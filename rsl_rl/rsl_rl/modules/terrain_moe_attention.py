"""Cross-attention MoE actor-critic for the Go2 terrain task.

The data path follows terrain_h_0729's ActorCriticChap3Attn:

    history -> K predictive experts -> Q/K/V attention -> fused latent
    actor([current proprio, velocity estimate, fused latent])

Online experts and the online fusion are trained by predictive losses. The
actor consumes a frozen EMA copy so PPO cannot leak gradients into the
representation through the actor.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .terrain_moe_actor_critic import (
    PredictiveExpert,
    build_contact_supervision_mask,
    build_mlp,
    masked_contact_bce,
)


class TerrainMoeAttentionActorCritic(nn.Module):
    is_recurrent = False
    is_terrain_moe = True
    architecture_version = "go2_terrain_moe_attn_v1"

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
        dynamics_target_dim=30,
        contact_dim=4,
        terrain_dim=187,
        activation="elu",
        init_noise_std=0.8,
        ema_tau=0.01,
        history_encoder="mlp",
        history_encoder_hidden_dims=(64, 64, 64),
        tcn_width=192,
        tcn_layers=3,
        velocity_hidden_dims=(128, 64),
        terrain_loss_coef=0.1,
        contact_loss_coef=1.0,
        dynamics_loss_coef=1.0,
        kl_loss_coef=0.1,
        velocity_loss_coef=1.0,
        fused_loss_coef=1.0,
        future_contact_steps=15,
        num_attn_heads=4,
        route_uniform_floor=0.1,
        attention_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        if latent_dim % num_attn_heads != 0:
            raise ValueError(
                "latent_dim must be divisible by num_attn_heads"
            )
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_proprio = num_proprio
        self.history_len = history_len
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.dynamics_target_dim = dynamics_target_dim
        self.contact_dim = contact_dim
        self.terrain_dim = terrain_dim
        self.ema_tau = ema_tau
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
        self.num_attn_heads = int(num_attn_heads)
        self.head_dim = latent_dim // self.num_attn_heads
        self.route_uniform_floor = float(route_uniform_floor)
        self.attention_temperature = float(attention_temperature)
        if not 0.0 <= self.route_uniform_floor <= 1.0:
            raise ValueError("route_uniform_floor must be in [0, 1]")
        if self.attention_temperature <= 0.0:
            raise ValueError("attention_temperature must be positive")
        self.fused_loss_coef = float(fused_loss_coef)
        contact_output_dim = contact_dim * future_contact_steps

        expert_kwargs = dict(
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
            history_encoder_hidden_dims=self.history_encoder_hidden_dims,
            tcn_width=tcn_width,
            tcn_layers=tcn_layers,
        )
        self.TsDyn_modules = nn.ModuleList(
            [PredictiveExpert(**expert_kwargs) for _ in range(num_modes)]
        )
        self.target_TsDyn_modules = copy.deepcopy(self.TsDyn_modules)

        self.velocity_estimator = build_mlp(
            num_actor_obs, list(velocity_hidden_dims), 3, activation
        )
        self.actor = build_mlp(
            num_proprio + latent_dim + 3,
            list(actor_hidden_dims),
            num_actions,
            activation,
        )
        self.critic = build_mlp(
            num_critic_obs, list(critic_hidden_dims), 1, activation
        )

        self.q_proj = nn.Linear(num_actor_obs, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.target_q_proj = copy.deepcopy(self.q_proj)
        self.target_k_proj = copy.deepcopy(self.k_proj)
        self.target_v_proj = copy.deepcopy(self.v_proj)
        self.target_out_proj = copy.deepcopy(self.out_proj)

        self.fused_decoder = build_mlp(
            latent_dim + num_actions,
            list(expert_hidden_dims),
            dynamics_target_dim,
            activation,
        )
        fused_hidden = list(expert_hidden_dims[:-1]) or [64]
        self.fused_contact_head = build_mlp(
            latent_dim + num_actions,
            fused_hidden,
            contact_output_dim,
            activation,
        )
        self.fused_terrain_head = build_mlp(
            latent_dim,
            fused_hidden,
            terrain_dim,
            activation,
        )

        for module in (
            self.target_TsDyn_modules,
            self.target_q_proj,
            self.target_k_proj,
            self.target_v_proj,
            self.target_out_proj,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.std = nn.Parameter(torch.ones(num_actions) * init_noise_std)
        self.distribution = None
        self.last_perception = None
        self.last_aux_metrics = {}
        Normal.set_default_validate_args = False

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def _encode_modes(self, history, experts):
        latents, logvars = [], []
        for expert in experts:
            mu, logvar = expert.encode(history)
            latents.append(mu)
            logvars.append(logvar)
        return torch.stack(latents, dim=1), torch.stack(logvars, dim=1)

    def _attend(self, history, expert_latents, projections):
        q_proj, k_proj, v_proj, out_proj = projections
        batch, modes, latent_dim = expert_latents.shape
        head_dim = self.head_dim
        query = q_proj(history).view(
            batch, self.num_attn_heads, head_dim
        )
        keys = k_proj(expert_latents).view(
            batch, modes, self.num_attn_heads, head_dim
        ).transpose(1, 2)
        values = v_proj(expert_latents).view(
            batch, modes, self.num_attn_heads, head_dim
        ).transpose(1, 2)
        logits = torch.einsum(
            "bhd,bhkd->bhk", query, keys
        ) / (head_dim ** 0.5)
        attention = F.softmax(
            logits / max(self.attention_temperature, 1e-6),
            dim=-1,
        )
        uniform = torch.full_like(attention, 1.0 / modes)
        attention = (
            (1.0 - self.route_uniform_floor) * attention
            + self.route_uniform_floor * uniform
        )
        fused = torch.einsum(
            "bhk,bhkd->bhd", attention, values
        ).reshape(batch, latent_dim)
        fused = out_proj(fused)
        return fused, attention.mean(dim=1)

    def _perception_from_modules(
        self, history, experts, projections, use_target
    ):
        expert_latents, expert_logvars = self._encode_modes(
            history, experts
        )
        fused, route = self._attend(
            history, expert_latents, projections
        )
        velocity = self.velocity_estimator(history).detach()
        base_latent = expert_latents.mean(dim=1)
        perception = {
            "expert_latents": expert_latents.detach(),
            "base_latent": base_latent.detach(),
            "control_latent": fused,
            "gate_weight": route,
            "residual": fused - base_latent.detach(),
            "velocity": velocity,
            "mode_mu": expert_latents,
            "mode_logvar": expert_logvars,
            "route": route,
            "use_target": use_target,
        }
        self.last_perception = perception
        return perception

    def perceive(self, history, use_target=True):
        if use_target:
            with torch.no_grad():
                return self._perception_from_modules(
                    history,
                    self.target_TsDyn_modules,
                    (
                        self.target_q_proj,
                        self.target_k_proj,
                        self.target_v_proj,
                        self.target_out_proj,
                    ),
                    True,
                )
        return self._perception_from_modules(
            history,
            self.TsDyn_modules,
            (self.q_proj, self.k_proj, self.v_proj, self.out_proj),
            False,
        )

    def _actor_input(self, history, perception):
        return torch.cat(
            (
                history[:, -self.num_proprio :],
                perception["control_latent"],
                perception["velocity"],
            ),
            dim=-1,
        )

    def _mean(self, history):
        perception = self.perceive(history, use_target=True)
        return self.actor(self._actor_input(history, perception)), perception

    def update_distribution(self, history):
        mean, perception = self._mean(history)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return mean, perception

    def act(self, history, **kwargs):
        mean, _ = self.update_distribution(history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def act_inference(self, history, **kwargs):
        mean, _ = self._mean(history)
        return mean

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

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
        valid = (
            torch.ones(
                history.size(0), dtype=torch.bool, device=history.device
            )
            if masks is None
            else masks.reshape(-1).bool()
        )
        if not valid.any():
            valid = torch.ones_like(valid)

        perception = self.perceive(history, use_target=False)
        mode_mu = perception["mode_mu"]
        mode_logvar = perception["mode_logvar"]
        contact_mask = build_contact_supervision_mask(
            history.size(0),
            self.future_contact_steps,
            self.contact_dim,
            valid,
            contact_valid,
            history.device,
            history.dtype,
        )
        contact_mask_sum = contact_mask.sum()

        dynamics_loss = history.new_zeros(())
        contact_loss = history.new_zeros(())
        terrain_loss = history.new_zeros(())
        for mode in range(self.num_modes):
            expert = self.TsDyn_modules[mode]
            sampled = expert.reparameterize(
                mode_mu[:, mode], mode_logvar[:, mode]
            )
            predicted_dynamics, contact_logits = expert.decode(
                sampled, actions
            )
            if contact_logits.size(-1) != (
                self.contact_dim * self.future_contact_steps
            ):
                raise RuntimeError(
                    "contact prediction dimension mismatch: "
                    f"{contact_logits.size(-1)}"
                )
            predicted_terrain = expert.terrain_head(mode_mu[:, mode])
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
        dynamics_loss = dynamics_loss / self.num_modes
        contact_loss = contact_loss / self.num_modes
        terrain_loss = terrain_loss / self.num_modes

        fused_latent = perception["control_latent"]
        fused_input = torch.cat((fused_latent, actions), dim=-1)
        fused_dynamics = self.fused_decoder(fused_input)
        fused_contact = self.fused_contact_head(fused_input)
        fused_terrain = self.fused_terrain_head(fused_latent)
        fused_dynamics_loss = F.mse_loss(
            fused_dynamics[valid], next_dynamics[valid]
        )
        fused_contact_loss = masked_contact_bce(
            fused_contact, next_contact, contact_mask
        )
        fused_terrain_loss = F.mse_loss(
            fused_terrain[valid], terrain_heights[valid]
        )

        kl_loss = (
            -0.5
            * (
                1.0
                + mode_logvar
                - mode_mu.pow(2)
                - mode_logvar.exp()
            )
            .mean(dim=-1)
            .mean()
        )
        velocity_loss = F.mse_loss(
            self.velocity_estimator(history)[valid],
            velocity_target[valid],
        )
        total_dynamics = (
            dynamics_loss + self.fused_loss_coef * fused_dynamics_loss
        )
        total_contact = (
            contact_loss + self.fused_loss_coef * fused_contact_loss
        )
        total_terrain = (
            terrain_loss + self.fused_loss_coef * fused_terrain_loss
        )
        total = (
            self.dynamics_loss_coef * total_dynamics
            + self.contact_loss_coef * total_contact
            + self.terrain_loss_coef * total_terrain
            + self.kl_loss_coef * kl_loss
            + self.velocity_loss_coef * velocity_loss
        )
        metrics = {
            "dynamics_loss": total_dynamics.detach(),
            "contact_loss": total_contact.detach(),
            "terrain_loss": total_terrain.detach(),
            "kl_loss": kl_loss.detach(),
            "velocity_loss": velocity_loss.detach(),
            "contact_valid_fraction": (
                contact_mask_sum
                / max(
                    history.size(0) * self.contact_dim
                    * self.future_contact_steps,
                    1,
                )
            ).detach(),
            "fused_dynamics_loss": fused_dynamics_loss.detach(),
            "fused_contact_loss": fused_contact_loss.detach(),
            "fused_terrain_loss": fused_terrain_loss.detach(),
        }
        self.last_aux_metrics = metrics
        return total, metrics

    @torch.no_grad()
    def update_ema_target(self):
        tau = self.ema_tau
        module_pairs = [
            (self.TsDyn_modules, self.target_TsDyn_modules),
            (self.q_proj, self.target_q_proj),
            (self.k_proj, self.target_k_proj),
            (self.v_proj, self.target_v_proj),
            (self.out_proj, self.target_out_proj),
        ]
        for online, target in module_pairs:
            if isinstance(online, nn.ModuleList):
                pairs = zip(online, target)
            else:
                pairs = ((online, target),)
            for online_module, target_module in pairs:
                for online_parameter, target_parameter in zip(
                    online_module.parameters(),
                    target_module.parameters(),
                ):
                    target_parameter.mul_(1.0 - tau)
                    target_parameter.add_(online_parameter, alpha=tau)

    def policy_parameters(self):
        return list(self.actor.parameters()) + list(
            self.critic.parameters()
        ) + [self.std]

    def perception_parameters(self):
        modules = [
            self.TsDyn_modules,
            self.velocity_estimator,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.fused_decoder,
            self.fused_contact_head,
            self.fused_terrain_head,
        ]
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]

    def architecture_metadata(self):
        return {
            "architecture_version": self.architecture_version,
            "num_modes": self.num_modes,
            "latent_dim": self.latent_dim,
            "history_len": self.history_len,
            "ema_tau": self.ema_tau,
            "future_contact_steps": self.future_contact_steps,
            "history_encoder": self.history_encoder,
            "history_encoder_hidden_dims": list(
                self.history_encoder_hidden_dims
            ),
            "num_attn_heads": self.num_attn_heads,
            "route_uniform_floor": self.route_uniform_floor,
            "attention_temperature": self.attention_temperature,
            "ppo_targets": ["actor", "critic"],
            "perception_targets": [
                "experts",
                "attention",
                "fused_decoders",
                "velocity",
            ],
        }
