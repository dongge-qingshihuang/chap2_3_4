import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import TerrainMoeStorage


class TerrainMoePPO:
    """PPO with a separate predictive-representation optimizer."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        perception_learning_rate=3e-4,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.perception_learning_rate = perception_learning_rate
        self.use_auxiliary_loss = getattr(
            actor_critic, "use_auxiliary_loss", True
        )

        self.optimizer = optim.Adam(
            actor_critic.policy_parameters(), lr=learning_rate
        )
        perception_parameters = actor_critic.perception_parameters()
        self.perception_optimizer = (
            optim.Adam(
                perception_parameters,
                lr=perception_learning_rate,
            )
            if perception_parameters
            else None
        )
        self.storage = None
        self.transition = TerrainMoeStorage.Transition()

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        dynamics_target_dim,
        contact_dim,
        terrain_dim,
        future_contact_steps,
    ):
        self.storage = TerrainMoeStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            dynamics_target_dim,
            contact_dim,
            terrain_dim,
            future_contact_steps,
            self.device,
        )

    def train_mode(self):
        self.actor_critic.train()

    def test_mode(self):
        self.actor_critic.eval()

    def act(self, observations, critic_observations):
        self.transition.observations = observations
        self.transition.critic_observations = critic_observations
        self.transition.actions = self.actor_critic.act(observations).detach()
        self.transition.values = self.actor_critic.evaluate(
            critic_observations
        ).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.mu = self.actor_critic.action_mean.detach()
        self.transition.sigma = self.actor_critic.action_std.detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, targets):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_dynamics = targets["dynamics"]
        self.transition.next_contact = targets["contact"].float()
        self.transition.terrain_heights = targets["terrain"]
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                dim=1,
            )
        self.storage.add_transitions(self.transition)
        self.transition = TerrainMoeStorage.Transition()

    def compute_returns(self, last_critic_observations):
        last_values = self.actor_critic.evaluate(
            last_critic_observations
        ).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        if self.use_auxiliary_loss:
            self.storage.build_future_contacts(
                self.actor_critic.future_contact_steps
            )
        policy_loss_sum = 0.0
        surrogate_loss_sum = 0.0
        value_loss_sum = 0.0
        perception_loss_sum = 0.0
        approx_kl_sum = 0.0
        clip_fraction_sum = 0.0
        entropy_sum = 0.0
        gate_entropy_sum = 0.0
        gate_temporal_norm_sum = 0.0
        gate_weight_deviation_sum = 0.0
        gate_tcn_grad_norm_sum = 0.0
        gate_temporal_pre_norm_sum = 0.0
        gate_balance_loss_sum = 0.0
        residual_norm_sum = 0.0
        base_norm_sum = 0.0
        gate_weight_sum = torch.zeros(
            self.actor_critic.num_modes, device=self.device
        )
        gate_weight_sq_sum = torch.zeros_like(gate_weight_sum)
        aux_sums = {}
        updates = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            old_values_batch,
            advantages_batch,
            returns_batch,
            old_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            next_dynamics,
            next_contact,
            contact_valid,
            terrain_heights,
            masks,
        ) in generator:
            self.actor_critic.act(obs_batch)
            log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            perception = self.actor_critic.last_perception
            with torch.no_grad():
                gate_weight = perception["gate_weight"]
                gate_prob = perception.get("gate_prob", gate_weight)
                gate_entropy = -(
                    gate_prob * (gate_prob + 1e-8).log()
                ).sum(dim=-1).mean()
                gate_entropy_sum += gate_entropy.item()
                gate_weight_sum += gate_weight.mean(dim=0)
                gate_weight_sq_sum += gate_weight.pow(2).mean(dim=0)
                gate_weight_deviation = perception.get(
                    "gate_weight_deviation",
                    gate_weight
                    - gate_weight.new_full(
                        gate_weight.shape, 1.0 / self.actor_critic.num_modes
                    ),
                )
                gate_weight_deviation_sum += (
                    gate_weight_deviation.norm(dim=-1).mean().item()
                )
                temporal_features = perception.get(
                    "temporal_gate_features"
                )
                if temporal_features is not None:
                    gate_temporal_norm_sum += (
                        temporal_features.norm(dim=-1).mean().item()
                    )
                temporal_pre_norm = perception.get(
                    "temporal_gate_pre_norm"
                )
                if temporal_pre_norm is not None:
                    gate_temporal_pre_norm_sum += (
                        temporal_pre_norm.item()
                    )
                residual_norm_sum += (
                    perception["residual"].norm(dim=-1).mean().item()
                )
                base_norm_sum += (
                    perception["base_latent"].norm(dim=-1).mean().item()
                )

            with torch.no_grad():
                log_ratio = torch.log(
                    sigma_batch / (old_sigma_batch + 1e-8) + 1e-8
                )
                approx_kl = (
                    log_ratio
                    + (
                        old_sigma_batch.pow(2)
                        + (old_mu_batch - mu_batch).pow(2)
                    )
                    / (2.0 * sigma_batch.pow(2) + 1e-8)
                    - 0.5
                ).sum(dim=-1).mean()

            ratio = torch.exp(
                log_prob_batch - old_log_prob_batch.squeeze(-1)
            )
            advantages = advantages_batch.squeeze(-1)
            surrogate = -advantages * ratio
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_clipped = -advantages * clipped_ratio
            surrogate_loss = torch.max(
                surrogate, surrogate_clipped
            ).mean()

            if self.use_clipped_value_loss:
                value_clipped = old_values_batch + (
                    value_batch - old_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            policy_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            gate_balance_coef = getattr(
                self.actor_critic, "gate_balance_coef", 0.0
            )
            if gate_balance_coef > 0.0:
                gate_usage = gate_prob.mean(dim=0)
                uniform_usage = gate_usage.new_full(
                    gate_usage.shape, 1.0 / gate_usage.numel()
                )
                gate_balance_loss = gate_usage.numel() * (
                    gate_usage - uniform_usage
                ).pow(2).sum()
                policy_loss = (
                    policy_loss
                    + gate_balance_coef * gate_balance_loss
                )
                gate_balance_loss_sum += gate_balance_loss.item()
            self.optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            temporal_gate = getattr(
                self.actor_critic, "temporal_gate_encoder", None
            )
            if temporal_gate is not None:
                temporal_sq = sum(
                    parameter.grad.pow(2).sum()
                    for parameter in temporal_gate.parameters()
                    if parameter.grad is not None
                )
                gate_tcn_grad_norm_sum += torch.sqrt(
                    temporal_sq
                ).item() if isinstance(temporal_sq, torch.Tensor) else 0.0
            nn.utils.clip_grad_norm_(
                self.actor_critic.policy_parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            if self.use_auxiliary_loss:
                predicted_velocity = critic_obs_batch[:, 45:48]
                perception_loss, aux_metrics = (
                    self.actor_critic.predictive_loss(
                        obs_batch,
                        actions_batch,
                        next_dynamics,
                        next_contact,
                        terrain_heights,
                        predicted_velocity,
                        contact_valid,
                        masks,
                    )
                )
                self.perception_optimizer.zero_grad(
                    set_to_none=True
                )
                perception_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.perception_parameters(),
                    self.max_grad_norm,
                )
                self.perception_optimizer.step()
            else:
                zero = obs_batch.new_zeros(())
                perception_loss = zero
                aux_metrics = {
                    "dynamics_loss": zero,
                    "contact_loss": zero,
                    "terrain_loss": zero,
                    "kl_loss": zero,
                    "velocity_loss": zero,
                    "contact_valid_fraction": zero,
                }

            if self.schedule == "adaptive" and self.desired_kl is not None:
                if approx_kl > 2.0 * self.desired_kl:
                    self.learning_rate = max(
                        1e-5, self.learning_rate / 1.5
                    )
                elif approx_kl < 0.5 * self.desired_kl:
                    self.learning_rate = min(
                        1e-2, self.learning_rate * 1.5
                    )
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

            policy_loss_sum += policy_loss.item()
            surrogate_loss_sum += surrogate_loss.item()
            value_loss_sum += value_loss.item()
            perception_loss_sum += perception_loss.item()
            approx_kl_sum += approx_kl.item()
            clip_fraction_sum += (
                (torch.abs(ratio - 1.0) > self.clip_param)
                .float()
                .mean()
                .item()
            )
            entropy_sum += entropy_batch.mean().item()
            for key, value in aux_metrics.items():
                aux_sums[key] = aux_sums.get(key, 0.0) + value.item()
            updates += 1

        self.storage.clear()
        self.actor_critic.update_ema_target()
        denominator = max(updates, 1)
        stats = {
            "policy_loss": policy_loss_sum / denominator,
            "value_loss": value_loss_sum / denominator,
            "surrogate_loss": surrogate_loss_sum / denominator,
            "perception_loss": perception_loss_sum / denominator,
            "approx_kl": approx_kl_sum / denominator,
            "clip_fraction": clip_fraction_sum / denominator,
            "entropy": entropy_sum / denominator,
            "gate_entropy": gate_entropy_sum / denominator,
            "gate_temporal_norm": gate_temporal_norm_sum / denominator,
            "gate_temporal_pre_norm": (
                gate_temporal_pre_norm_sum / denominator
            ),
            "gate_weight_deviation_norm": (
                gate_weight_deviation_sum / denominator
            ),
            "gate_tcn_grad_norm": gate_tcn_grad_norm_sum / denominator,
            "gate_balance_loss": gate_balance_loss_sum / denominator,
            "residual_norm": residual_norm_sum / denominator,
            "base_latent_norm": base_norm_sum / denominator,
        }
        mean_gate_weight = gate_weight_sum / denominator
        second_moment = gate_weight_sq_sum / denominator
        gate_std = (
            second_moment - mean_gate_weight.pow(2)
        ).clamp_min(0.0).sqrt()
        for mode in range(self.actor_critic.num_modes):
            stats[f"gate_weight_{mode}"] = mean_gate_weight[mode].item()
            stats[f"gate_weight_std_{mode}"] = gate_std[mode].item()
        for key, value in aux_sums.items():
            stats[key] = value / denominator
        return stats
