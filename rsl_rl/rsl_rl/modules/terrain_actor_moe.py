"""Standard actor-level mixture-of-experts policy.

Each expert is an action-producing MLP. A state-dependent gate computes a
convex combination of their actions. Both experts and the gate are trained
end-to-end by PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .terrain_moe_actor_critic import build_mlp


class MoEActor(nn.Module):
    def __init__(
        self,
        input_dim,
        num_actions,
        hidden_dims,
        activation,
        num_experts,
        gate_hidden_dims,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList(
            [
                build_mlp(
                    input_dim,
                    list(hidden_dims),
                    num_actions,
                    activation,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.gate = build_mlp(
            input_dim,
            list(gate_hidden_dims),
            self.num_experts,
            activation,
        )

    def forward(self, actor_input):
        gate_prob = F.softmax(self.gate(actor_input), dim=-1)
        expert_actions = torch.stack(
            [expert(actor_input) for expert in self.experts], dim=1
        )
        action = torch.einsum(
            "bk,bka->ba", gate_prob, expert_actions
        )
        return action, gate_prob, expert_actions


class TerrainActorMoEActorCritic(nn.Module):
    is_recurrent = False
    is_terrain_moe = True
    architecture_version = "go2_actor_moe_v1"
    gate_type = "actor_moe"
    temporal_gate_encoder = None
    gate_balance_coef = 0.0

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_proprio,
        num_recon=None,
        history_len=20,
        num_modes=4,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        gate_hidden_dims=(128, 64),
        velocity_hidden_dims=(128, 64),
        activation="elu",
        init_noise_std=1.0,
        dynamics_target_dim=30,
        contact_dim=4,
        terrain_dim=187,
        future_contact_steps=15,
        **kwargs,
    ):
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_proprio = num_proprio
        self.history_len = history_len
        self.num_modes = int(num_modes)
        self.dynamics_target_dim = dynamics_target_dim
        self.contact_dim = contact_dim
        self.terrain_dim = terrain_dim
        self.future_contact_steps = future_contact_steps
        self.velocity_estimator = build_mlp(
            num_actor_obs, list(velocity_hidden_dims), 3, activation
        )
        self.actor = MoEActor(
            input_dim=num_proprio + 3,
            num_actions=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            num_experts=self.num_modes,
            gate_hidden_dims=gate_hidden_dims,
        )
        self.critic = build_mlp(
            num_critic_obs, list(critic_hidden_dims), 1, activation
        )
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

    def actor_input(self, history):
        velocity = self.velocity_estimator(history).detach()
        return torch.cat(
            (history[:, -self.num_proprio :], velocity), dim=-1
        ), velocity

    def perceive(self, history, use_target=False):
        del use_target
        actor_input, velocity = self.actor_input(history)
        gate_prob = F.softmax(self.actor.gate(actor_input), dim=-1)
        zeros = gate_prob.new_zeros(gate_prob.size(0), 1)
        perception = {
            "velocity": velocity,
            "route": gate_prob,
            "gate_prob": gate_prob,
            "gate_weight": gate_prob,
            "gate_entropy": -(
                gate_prob * (gate_prob + 1e-8).log()
            ).sum(dim=-1).mean(),
            "gate_weight_deviation": gate_prob
            - gate_prob.new_full(
                gate_prob.shape, 1.0 / self.num_modes
            ),
            "temporal_gate_features": None,
            "base_latent": zeros,
            "control_latent": zeros,
            "residual": zeros,
        }
        self.last_perception = perception
        return perception

    def _mean(self, history):
        actor_input, velocity = self.actor_input(history)
        mean, gate_prob, expert_actions = self.actor(actor_input)
        perception = self.perceive(history)
        self.last_perception["gate_prob"] = gate_prob
        self.last_perception["gate_weight"] = gate_prob
        self.last_perception["route"] = gate_prob
        self.last_perception["expert_actions"] = expert_actions
        return mean, perception

    def update_distribution(self, history):
        mean, _ = self._mean(history)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return mean

    def act(self, history, **kwargs):
        self.update_distribution(history)
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
        del actions, next_dynamics, next_contact, terrain_heights
        del contact_valid
        valid = (
            torch.ones(
                history.size(0), dtype=torch.bool, device=history.device
            )
            if masks is None
            else masks.reshape(-1).bool()
        )
        if not valid.any():
            valid = torch.ones_like(valid)
        velocity_loss = F.mse_loss(
            self.velocity_estimator(history)[valid],
            velocity_target[valid],
        )
        zero = history.new_zeros(())
        metrics = {
            "dynamics_loss": zero.detach(),
            "contact_loss": zero.detach(),
            "terrain_loss": zero.detach(),
            "kl_loss": zero.detach(),
            "velocity_loss": velocity_loss.detach(),
            "contact_valid_fraction": history.new_ones(()),
        }
        self.last_aux_metrics = metrics
        return velocity_loss, metrics

    def update_ema_target(self):
        pass

    def policy_parameters(self):
        return (
            list(self.actor.parameters())
            + list(self.critic.parameters())
            + [self.std]
        )

    def perception_parameters(self):
        return list(self.velocity_estimator.parameters())

    def architecture_metadata(self):
        return {
            "architecture_version": self.architecture_version,
            "gate_type": self.gate_type,
            "num_experts": self.num_modes,
            "actor_hidden_dims": [
                layer.out_features
                for layer in self.actor.experts[0]
                if isinstance(layer, nn.Linear)
            ][:-1],
            "gate_hidden_dims": [
                layer.out_features
                for layer in self.actor.gate
                if isinstance(layer, nn.Linear)
            ][:-1],
            "history_len": self.history_len,
            "future_contact_steps": self.future_contact_steps,
            "ppo_targets": ["actor_experts", "actor_gate", "critic"],
            "perception_targets": ["velocity_estimator"],
        }
