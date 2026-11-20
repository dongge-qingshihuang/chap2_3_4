import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .load_aware_weights import LoadAwareWeightNetwork


def _activation(name):
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


def build_mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [nn.Linear(last_dim, hidden_dim), _activation(activation)()]
        )
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def build_contact_supervision_mask(
    batch_size,
    horizon,
    contact_dim,
    valid,
    contact_valid,
    device,
    dtype,
):
    if contact_valid is None:
        contact_valid = torch.ones(
            batch_size,
            horizon,
            dtype=torch.bool,
            device=device,
        )
    else:
        contact_valid = contact_valid.reshape(batch_size, horizon).bool()
    mask = contact_valid.to(dtype).repeat_interleave(contact_dim, dim=1)
    return mask * valid.to(dtype).unsqueeze(1)


def masked_contact_bce(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def zero_last_linear(module):
    last = next(
        layer
        for layer in reversed(list(module.modules()))
        if isinstance(layer, nn.Linear)
    )
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class MLPHistoryEncoder(nn.Module):
    """Flatten a history window and encode it with a narrow MLP.

    This mirrors the terrain_h_0729 expert encoder: it has no sequence-shaped
    convolution activations, so PPO updates retain far less memory.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        self.net = build_mlp(input_dim, list(hidden_dims), output_dim, activation)

    def forward(self, history):
        if history.dim() == 3:
            history = history.reshape(history.size(0), -1)
        return self.net(history)


class CausalTCNGateEncoder(nn.Module):
    """Lightweight stateless causal TCN for the control gate."""

    def __init__(
        self,
        num_proprio,
        history_len,
        channels=32,
        layers=2,
        temporal_dim=32,
        kernel_size=3,
        activation="elu",
        normalize_output=False,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("gate_tcn_layers must be at least 1")
        if kernel_size < 1:
            raise ValueError("gate_kernel_size must be positive")
        self.num_proprio = int(num_proprio)
        self.history_len = int(history_len)
        self.channels = int(channels)
        self.layers = int(layers)
        self.temporal_dim = int(temporal_dim)
        self.kernel_size = int(kernel_size)
        self.normalize_output = bool(normalize_output)
        self.activation = _activation(activation)()
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    self.num_proprio if layer == 0 else self.channels,
                    self.channels,
                    self.kernel_size,
                    dilation=2 ** layer,
                )
                for layer in range(self.layers)
            ]
        )
        for layer, convolution in enumerate(self.convolutions):
            left_padding = (
                self.kernel_size - 1
            ) * convolution.dilation[0]
            output_length = (
                self.history_len
                + left_padding
                - convolution.dilation[0] * (self.kernel_size - 1)
            )
            if output_length != self.history_len:
                raise ValueError(
                    f"gate TCN layer {layer} changes sequence length "
                    f"from {self.history_len} to {output_length}"
                )
        self.projection = nn.Linear(self.channels, self.temporal_dim)
        self.output_norm = (
            nn.LayerNorm(self.temporal_dim)
            if self.normalize_output
            else nn.Identity()
        )

    def forward_sequence(self, history, return_pre_norm=False):
        batch = history.size(0)
        sequence = history.reshape(
            batch, self.history_len, self.num_proprio
        ).transpose(1, 2)
        for convolution in self.convolutions:
            left_padding = (
                self.kernel_size - 1
            ) * convolution.dilation[0]
            sequence = F.pad(sequence, (left_padding, 0))
            sequence = self.activation(convolution(sequence))
        pre_norm = self.activation(
            self.projection(sequence.transpose(1, 2))
        )
        normalized = self.output_norm(pre_norm)
        if return_pre_norm:
            return normalized, pre_norm
        return normalized

    def forward(self, history, return_diagnostics=False):
        sequence, pre_norm = self.forward_sequence(
            history, return_pre_norm=True
        )
        features = sequence[:, -1]
        if not return_diagnostics:
            return features
        return features, {
            "pre_norm": pre_norm[:, -1].norm(dim=-1).mean(),
            "post_norm": features.norm(dim=-1).mean(),
        }


class PredictiveExpert(nn.Module):
    """Action-conditioned predictive expert used by the online and EMA paths."""

    def __init__(
        self,
        history_dim,
        history_len,
        action_dim,
        latent_dim,
        hidden_dims,
        dynamics_dim,
        contact_dim,
        terrain_dim,
        activation,
        history_encoder="mlp",
        history_encoder_hidden_dims=(64, 64, 64),
        tcn_width=192,
        tcn_layers=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_len = history_len
        self.proprio_dim = history_dim // history_len
        if self.proprio_dim * history_len != history_dim:
            raise ValueError("history_dim must be divisible by history_len")
        self.activation = _activation(activation)()
        self.history_encoder = str(history_encoder).lower()
        if self.history_encoder == "mlp":
            self.encoder = MLPHistoryEncoder(
                history_dim,
                2 * latent_dim,
                history_encoder_hidden_dims,
                activation,
            )
            self.encoder_fc = None
        elif self.history_encoder == "tcn":
            self.encoder = nn.ModuleList(
                [
                    nn.Conv1d(
                        self.proprio_dim if layer == 0 else tcn_width,
                        tcn_width,
                        3,
                        dilation=2 ** layer,
                        padding=2 ** layer,
                    )
                    for layer in range(tcn_layers)
                ]
            )
            self.encoder_fc = nn.Linear(tcn_width, 2 * latent_dim)
        else:
            raise ValueError(
                f"Unsupported history encoder: {self.history_encoder}"
            )
        self.decoder = build_mlp(
            latent_dim + action_dim,
            hidden_dims,
            dynamics_dim,
            activation,
        )
        self.contact_head = build_mlp(
            latent_dim + action_dim,
            hidden_dims[:-1] or [64],
            contact_dim,
            activation,
        )
        self.terrain_head = build_mlp(
            latent_dim, hidden_dims[:-1] or [64], terrain_dim, activation
        )

    def encode(self, history):
        if self.history_encoder == "mlp":
            encoded = self.encoder(history)
        else:
            sequence = history.view(
                history.size(0), self.history_len, self.proprio_dim
            ).transpose(1, 2)
            for convolution in self.encoder:
                sequence = self.activation(convolution(sequence))
            encoded = self.encoder_fc(sequence[:, :, -1])
        mu, logvar = encoded.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, latent, actions):
        decoder_input = torch.cat((latent, actions), dim=-1)
        dynamics = self.decoder(decoder_input)
        contact_logits = self.contact_head(decoder_input)
        return dynamics, contact_logits


class TerrainMoeActorCritic(nn.Module):
    """Predictive multi-expert actor with EMA-stabilized adaptive fusion.

    Online experts are trained only by predictive auxiliary losses. The actor
    consumes frozen EMA expert latents. PPO can update the actor, critic,
    control adapters, and state-dependent gate, but it cannot leak gradients
    into the predictive representation.
    """

    is_recurrent = False
    is_terrain_moe = True
    architecture_version = "go2_terrain_moe_v3_mlp"

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_proprio,
        num_recon=None,
        history_len=20,
        num_modes=4,
        latent_dim=32,
        expert_hidden_dims=(256, 128, 64),
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        gate_hidden_dims=(64, 64),
        dynamics_target_dim=30,
        contact_dim=4,
        terrain_dim=187,
        activation="elu",
        init_noise_std=0.8,
        min_noise_std=0.05,
        max_noise_std=0.5,
        ema_tau=0.01,
        gate_rho=0.3,
        gate_type="mlp",
        gate_tcn_channels=32,
        gate_tcn_layers=2,
        gate_temporal_dim=32,
        gate_kernel_size=3,
        gate_output_norm=False,
        gate_balance_coef=0.0,
        use_auxiliary_loss=True,
        train_perception_with_ppo=False,
        use_ema=True,
        history_encoder="mlp",
        history_encoder_hidden_dims=(64, 64, 64),
        tcn_width=192,
        tcn_layers=3,
        velocity_hidden_dims=(128, 64),
        adapter_hidden_dims=(64,),
        terrain_loss_coef=0.1,
        contact_loss_coef=1.0,
        dynamics_loss_coef=1.0,
        kl_loss_coef=0.1,
        velocity_loss_coef=1.0,
        future_contact_steps=15,
        num_critics=1,
        payload_dim=0,
        payload_hidden_dims=(256, 128),
        load_aware_enabled=False,
        load_weight_hidden_dim=64,
        load_weight_min=0.05,
        load_weight_max=0.90,
        base_action_residual_enabled=False,
        base_action_residual_context_dim=18,
        base_action_residual_hidden_dims=(128, 64),
        base_action_residual_clip=0.1,
        **kwargs,
    ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_proprio = num_proprio
        self.num_critics = int(num_critics)
        self.payload_dim = int(payload_dim)
        self.load_aware_enabled = bool(load_aware_enabled)
        self.base_action_residual_enabled = bool(
            base_action_residual_enabled
        )
        self.base_action_residual_context_dim = int(
            base_action_residual_context_dim
        )
        self.base_action_residual_feature_dim = (
            num_proprio + latent_dim + 3
        )
        self.base_action_residual_clip = float(
            base_action_residual_clip
        )
        if self.base_action_residual_clip <= 0:
            raise ValueError(
                "base_action_residual_clip must be positive"
            )
        self.finetune_mode = "partial"
        self.expert_freeze = True
        self.gate_lr_multiplier = 0.25
        self.adapter_lr_multiplier = 0.25
        self.actor_lr_multiplier = 1.0
        self.expert_lr_multiplier = 0.1
        self.history_len = history_len
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.noise_std_min = float(min_noise_std)
        self.noise_std_max = float(max_noise_std)
        if not (
            0.0 < self.noise_std_min
            < self.noise_std_max
        ):
            raise ValueError(
                "Require 0 < min_noise_std < max_noise_std, got "
                f"{self.noise_std_min} and {self.noise_std_max}"
            )
        if history_len * num_proprio != num_actor_obs:
            raise ValueError(
                "num_actor_obs must equal history_len * num_proprio: "
                f"{num_actor_obs} != {history_len} * {num_proprio}"
            )
        self.dynamics_target_dim = dynamics_target_dim
        self.contact_dim = contact_dim
        self.terrain_dim = terrain_dim
        self.ema_tau = ema_tau
        self.gate_rho = gate_rho
        self.gate_type = str(gate_type).lower()
        if self.gate_type not in ("mlp", "temporal_tcn", "uniform"):
            raise ValueError(f"Unsupported gate_type: {self.gate_type}")
        self.gate_tcn_channels = int(gate_tcn_channels)
        self.gate_tcn_layers = int(gate_tcn_layers)
        self.gate_temporal_dim = int(gate_temporal_dim)
        self.gate_kernel_size = int(gate_kernel_size)
        self.gate_output_norm = bool(gate_output_norm)
        self.gate_balance_coef = float(gate_balance_coef)
        self.use_auxiliary_loss = bool(use_auxiliary_loss)
        self.train_perception_with_ppo = bool(
            train_perception_with_ppo
        )
        self.use_ema = bool(use_ema)
        if self.train_perception_with_ppo and self.use_auxiliary_loss:
            raise ValueError(
                "train_perception_with_ppo=True requires "
                "use_auxiliary_loss=False"
            )
        self.history_encoder = str(history_encoder).lower()
        self.history_encoder_hidden_dims = tuple(
            int(dim) for dim in history_encoder_hidden_dims
        )
        self.tcn_width = tcn_width
        self.tcn_layers = tcn_layers
        self.terrain_loss_coef = terrain_loss_coef
        self.contact_loss_coef = contact_loss_coef
        self.dynamics_loss_coef = dynamics_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.future_contact_steps = future_contact_steps
        contact_output_dim = contact_dim * future_contact_steps

        self.TsDyn_modules = nn.ModuleList(
            [
                PredictiveExpert(
                    history_dim=num_actor_obs,
                    history_len=history_len,
                    action_dim=num_actions,
                    latent_dim=latent_dim,
                    hidden_dims=list(expert_hidden_dims),
                    dynamics_dim=dynamics_target_dim,
                    contact_dim=contact_output_dim,
                    terrain_dim=terrain_dim,
                    activation=activation,
                    history_encoder=self.history_encoder,
                    history_encoder_hidden_dims=(
                        self.history_encoder_hidden_dims
                    ),
                    tcn_width=tcn_width,
                    tcn_layers=tcn_layers,
                )
                for _ in range(num_modes)
            ]
        )
        self.target_TsDyn_modules = copy.deepcopy(self.TsDyn_modules)
        for parameter in self.target_TsDyn_modules.parameters():
            parameter.requires_grad_(False)

        self.velocity_estimator = build_mlp(
            num_actor_obs, list(velocity_hidden_dims), 3, activation
        )
        self.control_adapters = nn.ModuleList(
            [
                build_mlp(
                    latent_dim,
                    list(adapter_hidden_dims),
                    latent_dim,
                    activation,
                )
                for _ in range(num_modes)
            ]
        )
        for adapter in self.control_adapters:
            zero_last_linear(adapter)

        self.temporal_gate_encoder = (
            CausalTCNGateEncoder(
                num_proprio,
                history_len,
                channels=self.gate_tcn_channels,
                layers=self.gate_tcn_layers,
                temporal_dim=self.gate_temporal_dim,
                kernel_size=self.gate_kernel_size,
                activation=activation,
                normalize_output=self.gate_output_norm,
            )
            if self.gate_type == "temporal_tcn"
            else None
        )
        if self.gate_type == "uniform":
            self.control_gate = None
            self.architecture_version = (
                "go2_terrain_moe_v3_mlp_uniform_routing"
            )
        else:
            gate_input_dim = (
                num_proprio + num_modes * latent_dim
                + (
                    self.gate_temporal_dim
                    if self.temporal_gate_encoder is not None
                    else 0
                )
            )
            self.control_gate = build_mlp(
                gate_input_dim,
                list(gate_hidden_dims),
                num_modes,
                activation,
            )
            zero_last_linear(self.control_gate)
        if self.gate_type == "temporal_tcn":
            suffix = "_norm" if self.gate_output_norm else ""
            self.architecture_version = (
                f"go2_terrain_moe_v3_mlp_temporal_tcn_gate{suffix}"
            )
        if not self.use_ema:
            self.architecture_version += "_no_ema"
        if self.train_perception_with_ppo:
            self.architecture_version += "_ppo_only"

        self.actor = build_mlp(
            num_proprio + latent_dim + 3,
            list(actor_hidden_dims),
            num_actions,
            activation,
        )
        if self.num_critics == 1:
            self.critic = build_mlp(
                num_critic_obs, list(critic_hidden_dims), 1, activation
            )
            self.critic_trunk = None
            self.critic_heads = None
        else:
            critic_layers = []
            last_dim = num_critic_obs
            for hidden_dim in critic_hidden_dims:
                critic_layers.extend(
                    [nn.Linear(last_dim, hidden_dim), _activation(activation)()]
                )
                last_dim = hidden_dim
            self.critic = None
            self.critic_trunk = nn.Sequential(*critic_layers)
            self.critic_heads = nn.ModuleList(
                [nn.Linear(last_dim, 1) for _ in range(self.num_critics)]
            )
        self.payload_estimator = (
            build_mlp(
                num_actor_obs,
                list(payload_hidden_dims),
                self.payload_dim,
                activation,
            )
            if self.payload_dim > 0
            else None
        )
        self.base_action_residual = None
        if self.base_action_residual_enabled:
            self.base_action_residual = build_mlp(
                self.base_action_residual_feature_dim
                + self.base_action_residual_context_dim,
                list(base_action_residual_hidden_dims),
                num_actions,
                activation,
            )
            zero_last_linear(self.base_action_residual)
        self.load_weight_net = (
            LoadAwareWeightNetwork(
                payload_dim=self.payload_dim,
                hidden_dim=load_weight_hidden_dim,
                min_weight=load_weight_min,
                max_weight=load_weight_max,
                num_critics=self.num_critics,
            )
            if self.load_aware_enabled
            else None
        )
        if self.load_aware_enabled:
            if self.payload_dim != 9:
                raise ValueError(
                    "load_aware_enabled=True requires a 9-D payload estimator"
                )
            if self.num_critics != 3:
                raise ValueError(
                    "load_aware_enabled=True requires three critics"
                )
        initial_std = min(
            max(float(init_noise_std), self.noise_std_min),
            self.noise_std_max,
        )
        self.log_std = nn.Parameter(
            torch.full((num_actions,), math.log(initial_std))
        )
        self.distribution = None
        self.last_perception = None
        self.last_payload_prediction = None
        self.last_load_weights = None
        self.last_base_action_mean = None
        self.last_action_residual = None
        self.last_aux_metrics = {}
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def std(self):
        return self.log_std.exp()

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    @torch.no_grad()
    def enforce_std_bounds(self):
        lower = math.log(self.noise_std_min)
        upper = math.log(self.noise_std_max)
        clamped = (self.log_std < lower) | (self.log_std > upper)
        self.log_std.clamp_(lower, upper)
        return clamped.float().mean()

    def load_state_dict(self, state_dict, strict=True):
        """Load old ``std`` checkpoints into the bounded ``log_std`` parameter."""
        state_dict = OrderedDict(state_dict)
        if "std" in state_dict:
            old_std = state_dict.pop("std")
            if "log_std" not in state_dict:
                bounded_std = old_std.detach().clamp(
                    min=self.noise_std_min,
                    max=self.noise_std_max,
                )
                state_dict["log_std"] = bounded_std.log()
        return super().load_state_dict(state_dict, strict=strict)

    def _online_latents(self, history):
        latents = []
        for expert in self.TsDyn_modules:
            mu, _ = expert.encode(history)
            latents.append(mu)
        return torch.stack(latents, dim=1)

    @torch.no_grad()
    def _target_latents(self, history):
        latents = []
        for expert in self.target_TsDyn_modules:
            mu, _ = expert.encode(history)
            latents.append(mu)
        return torch.stack(latents, dim=1)

    def perceive(self, history, use_target=True):
        proprio = history[:, -self.num_proprio :]
        if use_target:
            expert_latents = self._target_latents(history)
        else:
            expert_latents = self._online_latents(history)
            if not self.train_perception_with_ppo:
                expert_latents = expert_latents.detach()

        base_latent = expert_latents.mean(dim=1)
        normalized_expert_latents = F.normalize(
            expert_latents, dim=-1
        )
        if self.control_gate is None:
            gate_prob = expert_latents.new_full(
                (history.size(0), self.num_modes),
                1.0 / self.num_modes,
            )
            temporal_features = None
            temporal_diag = None
        else:
            gate_inputs = [
                proprio,
                normalized_expert_latents.flatten(1),
            ]
            if self.temporal_gate_encoder is not None:
                temporal_features, temporal_diag = (
                    self.temporal_gate_encoder(
                        history, return_diagnostics=True
                    )
                )
                gate_inputs.append(temporal_features)
            else:
                temporal_features = None
                temporal_diag = None
            gate_input = torch.cat(gate_inputs, dim=-1)
            gate_prob = torch.softmax(
                self.control_gate(gate_input), dim=-1
            )
        gate_weight = (
            (1.0 - self.gate_rho) / self.num_modes
            + self.gate_rho * gate_prob
        )

        adapted = torch.stack(
            [
                adapter(F.normalize(expert_latents[:, index], dim=-1))
                for index, adapter in enumerate(self.control_adapters)
            ],
            dim=1,
        )
        residual = (gate_weight.unsqueeze(-1) * adapted).sum(dim=1)
        if self.train_perception_with_ppo:
            control_latent = base_latent + residual
            velocity = self.velocity_estimator(history)
            diagnostic_expert_latents = expert_latents
            diagnostic_base_latent = base_latent
        else:
            control_latent = base_latent.detach() + residual
            velocity = self.velocity_estimator(history).detach()
            diagnostic_expert_latents = expert_latents.detach()
            diagnostic_base_latent = base_latent.detach()
        uniform_weight = gate_weight.new_full(
            gate_weight.shape, 1.0 / self.num_modes
        )
        gate_entropy = -(
            gate_prob * (gate_prob + 1e-8).log()
        ).sum(dim=-1).mean()

        payload_prediction = (
            self.payload_estimator(history)
            if self.payload_estimator is not None
            else None
        )
        if (
            self.load_weight_net is not None
            and payload_prediction is not None
        ):
            load_weights = self.load_weight_net(
                payload_prediction.detach()
            )
        else:
            load_weights = None
        self.last_payload_prediction = payload_prediction
        self.last_load_weights = load_weights

        perception = {
            "expert_latents": diagnostic_expert_latents,
            "base_latent": diagnostic_base_latent,
            "control_latent": control_latent,
            "gate_weight": gate_weight,
            "gate_prob": gate_prob,
            "gate_entropy": gate_entropy,
            "gate_weight_deviation": gate_weight - uniform_weight,
            "temporal_gate_features": temporal_features,
            "temporal_gate_pre_norm": (
                temporal_diag["pre_norm"]
                if temporal_diag is not None
                else None
            ),
            "temporal_gate_post_norm": (
                temporal_diag["post_norm"]
                if temporal_diag is not None
                else None
            ),
            "residual": residual,
            "velocity": velocity,
            "payload_prediction": payload_prediction,
            "load_weights": load_weights,
        }
        self.last_perception = perception
        return perception

    def _actor_input(self, history, perception):
        proprio = history[:, -self.num_proprio :]
        return torch.cat(
            (
                proprio,
                perception["control_latent"],
                perception["velocity"],
            ),
            dim=-1,
        )

    def _mean(self, history, residual_context=None):
        perception = self.perceive(
            history,
            use_target=self.use_ema
            and not self.train_perception_with_ppo,
        )
        actor_input = self._actor_input(history, perception)
        base_action_mean = self.actor(actor_input)
        self.last_base_action_mean = base_action_mean
        if self.base_action_residual is None:
            self.last_action_residual = None
            return base_action_mean, perception
        if residual_context is None:
            raise ValueError(
                "Residual context is required when "
                "base_action_residual_enabled=True"
            )
        if residual_context.size(-1) != self.base_action_residual_context_dim:
            raise ValueError(
                "Residual context dimension mismatch: "
                f"{residual_context.size(-1)} != "
                f"{self.base_action_residual_context_dim}"
            )
        residual_input = torch.cat(
            (
                actor_input.detach(),
                residual_context.to(
                    device=history.device, dtype=history.dtype
                ),
            ),
            dim=-1,
        )
        action_residual = self.base_action_residual_clip * torch.tanh(
            self.base_action_residual(residual_input)
        )
        self.last_action_residual = action_residual
        return base_action_mean + action_residual, perception

    def update_distribution(self, history, residual_context=None):
        mean, _ = self._mean(
            history, residual_context=residual_context
        )
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return mean

    def act(self, history, residual_context=None, **kwargs):
        self.update_distribution(
            history, residual_context=residual_context
        )
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def act_inference(self, history, residual_context=None, **kwargs):
        mean, _ = self._mean(
            history, residual_context=residual_context
        )
        return mean

    def evaluate(self, critic_observations, **kwargs):
        if self.critic is not None:
            return self.critic(critic_observations)
        features = self.critic_trunk(critic_observations)
        return torch.cat(
            [head(features) for head in self.critic_heads], dim=-1
        )

    def predictive_loss(
        self,
        history,
        actions,
        next_dynamics,
        next_contact,
        terrain_heights,
        velocity_target,
        contact_valid=None,
        masks=None,
    ):
        dynamics_loss = history.new_zeros(())
        contact_loss = history.new_zeros(())
        terrain_loss = history.new_zeros(())
        kl_loss = history.new_zeros(())
        valid = (
            torch.ones(
                history.size(0), dtype=torch.bool, device=history.device
            )
            if masks is None
            else masks.reshape(-1).bool()
        )
        if not valid.any():
            valid = torch.ones_like(valid)

        batch_size = history.size(0)
        horizon = self.future_contact_steps
        contact_output_dim = self.contact_dim * horizon
        contact_mask = build_contact_supervision_mask(
            batch_size,
            horizon,
            self.contact_dim,
            valid,
            contact_valid,
            history.device,
            history.dtype,
        )
        contact_mask_sum = contact_mask.sum()

        for expert in self.TsDyn_modules:
            mu, logvar = expert.encode(history)
            sampled = expert.reparameterize(mu, logvar)
            predicted_dynamics, contact_logits = expert.decode(sampled, actions)
            if contact_logits.size(-1) != contact_output_dim:
                raise RuntimeError(
                    f"contact prediction dimension {contact_logits.size(-1)} "
                    f"does not match {contact_output_dim}"
                )
            predicted_terrain = expert.terrain_head(mu)
            dynamics_loss = dynamics_loss + F.mse_loss(
                predicted_dynamics[valid],
                next_dynamics[valid],
            )
            contact_loss = contact_loss + masked_contact_bce(
                contact_logits,
                next_contact,
                contact_mask,
            )
            terrain_loss = terrain_loss + F.mse_loss(
                predicted_terrain[valid],
                terrain_heights[valid],
            )
            kl_loss = kl_loss + (
                -0.5
                * (1.0 + logvar - mu.pow(2) - logvar.exp())
                .mean(dim=-1)
                .mean()
            )

        dynamics_loss = dynamics_loss / self.num_modes
        contact_loss = contact_loss / self.num_modes
        terrain_loss = terrain_loss / self.num_modes
        kl_loss = kl_loss / self.num_modes
        velocity_loss = F.mse_loss(
            self.velocity_estimator(history)[valid],
            velocity_target[valid],
        )
        total = (
            self.dynamics_loss_coef * dynamics_loss
            + self.contact_loss_coef * contact_loss
            + self.terrain_loss_coef * terrain_loss
            + self.kl_loss_coef * kl_loss
            + self.velocity_loss_coef * velocity_loss
        )
        metrics = {
            "dynamics_loss": dynamics_loss.detach(),
            "contact_loss": contact_loss.detach(),
            "terrain_loss": terrain_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "velocity_loss": velocity_loss.detach(),
            "contact_valid_fraction": (
                contact_mask_sum
                / max(batch_size * contact_output_dim, 1)
            ).detach(),
        }
        self.last_aux_metrics = metrics
        return total, metrics

    @torch.no_grad()
    def update_ema_target(self):
        if not self.use_auxiliary_loss or not self.use_ema:
            return
        tau = self.ema_tau
        for online, target in zip(
            self.TsDyn_modules, self.target_TsDyn_modules
        ):
            for online_parameter, target_parameter in zip(
                online.parameters(), target.parameters()
            ):
                target_parameter.mul_(1.0 - tau)
                target_parameter.add_(online_parameter, alpha=tau)

    def configure_finetune(
        self,
        mode="partial",
        expert_freeze=True,
        gate_lr_multiplier=0.25,
        adapter_lr_multiplier=0.25,
        actor_lr_multiplier=1.0,
        expert_lr_multiplier=0.1,
        freeze_actor=False,
    ):
        """Apply real parameter freezing for Chapter 4 PPO fine-tuning."""
        mode = str(mode).lower()
        if mode not in ("partial", "full"):
            raise ValueError(
                f"pretrained_policy_finetune_mode must be partial/full, got {mode}"
            )
        self.finetune_mode = mode
        self.expert_freeze = bool(expert_freeze)
        self.gate_lr_multiplier = float(gate_lr_multiplier)
        self.adapter_lr_multiplier = float(adapter_lr_multiplier)
        self.actor_lr_multiplier = float(actor_lr_multiplier)
        self.expert_lr_multiplier = float(expert_lr_multiplier)
        expert_requires_grad = mode == "full" and not self.expert_freeze
        self.train_perception_with_ppo = expert_requires_grad
        for parameter in self.target_TsDyn_modules.parameters():
            parameter.requires_grad_(False)
        for parameter in self.TsDyn_modules.parameters():
            parameter.requires_grad_(expert_requires_grad)
        # The pretrained velocity estimator and online experts are frozen in
        # partial mode. Full mode opts into PPO gradients for them.
        for parameter in self.velocity_estimator.parameters():
            parameter.requires_grad_(expert_requires_grad)
        for module in (
            self.actor,
            self.control_adapters,
            self.critic_trunk,
            self.critic_heads,
            self.critic,
            self.control_gate,
            self.temporal_gate_encoder,
            self.payload_estimator,
            self.load_weight_net,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.log_std.requires_grad_(True)
        if freeze_actor:
            for module in (
                self.actor,
                self.control_adapters,
                self.control_gate,
                self.temporal_gate_encoder,
            ):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
            self.log_std.requires_grad_(False)
        return self

    def get_payload_estimate(self):
        return self.last_payload_prediction

    def get_load_weights(self):
        return self.last_load_weights

    def payload_parameters(self):
        if self.payload_estimator is None:
            return []
        return list(self.payload_estimator.parameters())

    def load_gate_parameters(self):
        if self.load_weight_net is None:
            return []
        return list(self.load_weight_net.parameters())

    def policy_parameter_groups(self):
        groups = []
        if self.base_action_residual is not None:
            groups.append(
                {
                    "params": list(
                        self.base_action_residual.parameters()
                    ),
                    "lr_multiplier": self.actor_lr_multiplier,
                    "name": "base_action_residual",
                }
            )
        groups.extend(
            [
                {
                    "params": list(self.actor.parameters())
                    + [self.log_std],
                    "lr_multiplier": self.actor_lr_multiplier,
                    "name": "actor",
                },
                {
                    "params": list(
                        self.control_adapters.parameters()
                    ),
                    "lr_multiplier": self.adapter_lr_multiplier,
                    "name": "control_adapters",
                },
            ]
        )
        if self.control_gate is not None:
            gate_parameters = list(self.control_gate.parameters())
            if self.temporal_gate_encoder is not None:
                gate_parameters += list(
                    self.temporal_gate_encoder.parameters()
                )
            groups.append(
                {
                    "params": gate_parameters,
                    "lr_multiplier": self.gate_lr_multiplier,
                    "name": "gate",
                }
            )
        critic = (
            self.critic
            if self.critic is not None
            else self.critic_heads
        )
        critic_parameters = list(critic.parameters())
        if self.critic_trunk is not None:
            critic_parameters += list(self.critic_trunk.parameters())
        groups.append(
            {
                "params": critic_parameters,
                "lr_multiplier": 1.0,
                "name": "critic",
            }
        )
        if self.finetune_mode == "full" and not self.expert_freeze:
            groups.extend(
                [
                    {
                        "params": list(self.TsDyn_modules.parameters()),
                        "lr_multiplier": self.expert_lr_multiplier,
                        "name": "online_experts",
                    },
                    {
                        "params": list(self.velocity_estimator.parameters()),
                        "lr_multiplier": self.gate_lr_multiplier,
                        "name": "velocity_estimator",
                    },
                ]
            )
        return [
            group
            for group in groups
            if any(parameter.requires_grad for parameter in group["params"])
        ]

    def ppo_parameters(self):
        return [
            parameter
            for group in self.policy_parameter_groups()
            for parameter in group["params"]
        ]

    def perception_parameters(self):
        if self.train_perception_with_ppo:
            return []
        modules = [self.TsDyn_modules, self.velocity_estimator]
        return [
            parameter for module in modules for parameter in module.parameters()
        ]

    def architecture_metadata(self):
        targets = ["actor", "critic", "adapters"]
        if self.control_gate is not None:
            targets.insert(2, "gate")
        if self.finetune_mode == "full" and not self.expert_freeze:
            targets.extend(["online_experts", "velocity_estimator"])
        perception_targets = []
        if self.payload_estimator is not None:
            perception_targets.append("payload_estimator")
        if self.load_weight_net is not None:
            perception_targets.append("load_weight_net")
        metadata = {
            "architecture_version": self.architecture_version,
            "num_modes": self.num_modes,
            "latent_dim": self.latent_dim,
            "history_len": self.history_len,
            "ema_tau": self.ema_tau,
            "gate_rho": self.gate_rho,
            "gate_type": self.gate_type,
            "gate_tcn_channels": self.gate_tcn_channels,
            "gate_tcn_layers": self.gate_tcn_layers,
            "gate_temporal_dim": self.gate_temporal_dim,
            "gate_kernel_size": self.gate_kernel_size,
            "gate_output_norm": self.gate_output_norm,
            "gate_balance_coef": self.gate_balance_coef,
            "use_auxiliary_loss": self.use_auxiliary_loss,
            "train_perception_with_ppo": (
                self.train_perception_with_ppo
            ),
            "use_ema": self.use_ema,
            "finetune_mode": self.finetune_mode,
            "num_critics": self.num_critics,
            "action_noise_parameter": "log_std",
            "noise_std_min": self.noise_std_min,
            "noise_std_max": self.noise_std_max,
            "payload_dim": self.payload_dim,
            "load_aware_enabled": self.load_aware_enabled,
            "future_contact_steps": self.future_contact_steps,
            "history_encoder": self.history_encoder,
            "history_encoder_hidden_dims": list(
                self.history_encoder_hidden_dims
            ),
            "tcn_width": self.tcn_width,
            "tcn_layers": self.tcn_layers,
            "ppo_targets": targets,
            "perception_targets": (
                perception_targets
            ),
        }
        if self.base_action_residual_enabled:
            metadata.update(
                {
                    "base_action_residual_enabled": True,
                    "base_action_residual_context_dim": (
                        self.base_action_residual_context_dim
                    ),
                    "base_action_residual_feature_dim": (
                        self.base_action_residual_feature_dim
                    ),
                    "base_action_residual_clip": (
                        self.base_action_residual_clip
                    ),
                }
            )
        return metadata
