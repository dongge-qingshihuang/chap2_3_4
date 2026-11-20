# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage
from rsl_rl.modules import ActorCritic_dream
# from rsl_rl.modules import ActorCritic_woattention
from rsl_rl.modules import ActorCritic_attention
from rsl_rl.modules import ActorCritic_attention1
from rsl_rl.modules import ActorCriticArmServo
from rsl_rl.modules import ActorCriticArmServoDogPrior
from rsl_rl.modules import ActorCritic
from rsl_rl.modules import (
    payload_prediction_errors,
    physical_prior_weights,
    weight_entropy,
)

class HIMPPO:
    actor_critic: ActorCriticArmServoDogPrior
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 aux_loss_coef=0.1,
                 entropy_coef_final=None,
                 entropy_anneal_start_frac=0.25,
                 value_dim=1,
                 lambda_task=0.7,
                 lambda_stab=0.2,
                 lambda_energy=0.1,
                 load_aware_enabled=False,
                 payload_loss_coef=1.0,
                 load_weight_supervision_coef=1.0,
                 load_weight_smooth_coef=0.01,
                 load_gate_learning_rate=3e-4,
                 payload_learning_rate=3e-4,
                 load_risk_position_coef=1.0,
                 load_risk_torque_coef=1.0,
                 load_risk_velocity_coef=1.0,
                 load_stab_weight_min=0.20,
                 load_stab_weight_max=0.70,
                 load_risk_threshold=0.5,
                 load_risk_slope=5.0,
                 task_energy_ratio=0.8,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.aux_loss_coef = aux_loss_coef

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.transition = HIMRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        # entropy 退火：前 entropy_anneal_start_frac 比例的迭代恒定初值，
        # 之后线性降到 entropy_coef_final。final=None 时关闭退火（保持原行为）。
        self.entropy_coef_init = entropy_coef
        self.entropy_coef_final = entropy_coef_final
        self.entropy_anneal_start_frac = entropy_anneal_start_frac
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # 多评论家：三路 advantage / value loss 的加权系数
        self.value_dim = int(value_dim)
        if self.value_dim > 1:
            self.value_weights = torch.tensor(
                [lambda_task, lambda_stab, lambda_energy][:self.value_dim], device=self.device
            )
            self.value_weights = self.value_weights / self.value_weights.sum()
        else:
            self.value_weights = None

        self.load_aware_enabled = bool(load_aware_enabled)
        self.load_weight_supervision_coef = float(
            load_weight_supervision_coef
        )
        self.load_weight_smooth_coef = float(load_weight_smooth_coef)
        self.load_risk_position_coef = float(load_risk_position_coef)
        self.load_risk_torque_coef = float(load_risk_torque_coef)
        self.load_risk_velocity_coef = float(load_risk_velocity_coef)
        self.load_stab_weight_min = float(load_stab_weight_min)
        self.load_stab_weight_max = float(load_stab_weight_max)
        self.load_risk_threshold = float(load_risk_threshold)
        self.load_risk_slope = float(load_risk_slope)
        self.task_energy_ratio = float(task_energy_ratio)
        self.last_update_stats = {}

        if hasattr(actor_critic, "policy_parameter_groups"):
            parameter_groups = []
            for group in actor_critic.policy_parameter_groups():
                parameter_groups.append(
                    {
                        "params": group["params"],
                        "lr": learning_rate
                        * float(group.get("lr_multiplier", 1.0)),
                        "lr_multiplier": float(
                            group.get("lr_multiplier", 1.0)
                        ),
                        "name": group.get("name", "policy"),
                    }
                )
            self.optimizer = optim.Adam(parameter_groups, lr=learning_rate)
        else:
            ppo_parameters = (
                actor_critic.ppo_parameters()
                if hasattr(actor_critic, "ppo_parameters")
                else actor_critic.parameters()
            )
            self.optimizer = optim.Adam(ppo_parameters, lr=learning_rate)
        payload_parameters = (
            actor_critic.payload_parameters()
            if hasattr(actor_critic, "payload_parameters")
            else []
        )
        self.payload_optimizer = (
            optim.Adam(payload_parameters, lr=payload_learning_rate)
            if payload_parameters
            else None
        )
        load_gate_parameters = (
            actor_critic.load_gate_parameters()
            if hasattr(actor_critic, "load_gate_parameters")
            else []
        )
        self.load_gate_optimizer = (
            optim.Adam(load_gate_parameters, lr=load_gate_learning_rate)
            if load_gate_parameters
            else None
        )
        if self.load_aware_enabled and self.load_gate_optimizer is None:
            raise ValueError(
                "load_aware_enabled=True requires an actor load-weight network"
            )
        if self.load_aware_enabled and self.value_dim != 3:
            raise ValueError(
                "load-aware advantage fusion requires three critics"
            )

        # --- 第二章负载感知：黑盒估计器的监督（特权真值 -> ProprioEncoder 预测）---
        # 由 runner 从 env 写入；None 表示未启用
        self.payload_slice = None
        self.payload_loss_coef = float(payload_loss_coef)

        # --- 底座顺应残差（联合 fine-tune）---
        self.base_residual = None
        self.base_optimizer = None
        self.train_base = False
        self.base_learning_rate = 2e-4
        self.base_entropy_coef = 0.005
        self.base_log_std_max = 0.5

    def set_base_residual(self, base_residual, learning_rate=2e-4, entropy_coef=0.005,
                          entropy_coef_final=None, entropy_anneal_start_frac=0.25):
        """注册可训练的底座残差网络，并为其建立独立的低学习率 optimizer。"""
        self.base_residual = base_residual
        self.base_learning_rate = learning_rate
        self.base_entropy_coef = entropy_coef
        # 残差 entropy 退火：防止 std 在熵项推动下持续膨胀、退化成噪声源
        self.base_entropy_coef_init = entropy_coef
        self.base_entropy_coef_final = entropy_coef_final
        self.base_entropy_anneal_start_frac = entropy_anneal_start_frac
        self.base_log_std_max = float(getattr(base_residual, "std_max", 0.5))
        self.base_optimizer = optim.Adam(self.base_residual.parameters(), lr=learning_rate)
        self.train_base = True

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
                     base_obs_shape=None, base_actions_shape=None,
                     residual_context_shape=None):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device,
                                         base_obs_shape=base_obs_shape, base_actions_shape=base_actions_shape,
                                         residual_context_shape=residual_context_shape,
                                         value_dim=self.value_dim)

    def test_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, residual_context=None):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(
            obs, residual_context=residual_context
        ).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.residual_context = (
            residual_context.detach().clone()
            if residual_context is not None
            else None
        )
        if self.value_dim > 1:
            get_payload = getattr(
                self.actor_critic, "get_payload_estimate", None
            )
            get_weights = getattr(
                self.actor_critic, "get_load_weights", None
            )
            payload_prediction = (
                get_payload() if get_payload is not None else None
            )
            load_weights = (
                get_weights() if get_weights is not None else None
            )
            self.transition.payload_predictions = (
                payload_prediction.detach()
                if payload_prediction is not None
                else None
            )
            self.transition.load_weights = (
                load_weights.detach()
                if load_weights is not None
                else None
            )
            if self.load_aware_enabled and self.transition.load_weights is None:
                raise RuntimeError(
                    "Load-aware rollout did not produce fusion weights"
                )
        # need to record obs and critic_obs before env.step()
        # env.step() may reset and zero obs_buf in-place. Keep independent
        # pre-action observations for this transition.
        self.transition.observations = obs.detach().clone()
        self.transition.critic_observations = critic_obs.detach().clone()
        return self.transition.actions

    def act_base(self, base_obs):
        """采样底座残差动作并记录到 transition（与 arm 共享同一 reward/advantage）。"""
        residual = self.base_residual.act(base_obs).detach()
        self.transition.base_actions = residual
        self.transition.base_actions_log_prob = self.base_residual.get_actions_log_prob(residual).detach()
        self.transition.base_action_mean = self.base_residual.action_mean.detach()
        self.transition.base_action_sigma = self.base_residual.action_std.detach()
        self.transition.base_observations = base_obs
        return residual
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        time_outs = infos.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(dones, dtype=torch.bool).reshape(-1)
        else:
            time_outs = time_outs.to(self.device).bool().reshape(-1)
        self.transition.timeouts = time_outs
        if time_outs.any():
            self.transition.timeout_bootstrap_values = (
                self.actor_critic.evaluate(next_critic_obs).detach()
            )
        else:
            self.transition.timeout_bootstrap_values = torch.zeros_like(
                self.transition.values
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        if self.value_dim > 1:
            self.storage.compute_fused_advantages(self.value_weights)

    @staticmethod
    def _anneal_linear(it, total_it, init_val, final_val, start_frac):
        """前 start_frac 段恒定初值，之后线性降到 final。"""
        start_it = start_frac * total_it
        if it <= start_it:
            return init_val
        if it >= total_it:
            return final_val
        frac = (it - start_it) / max(total_it - start_it, 1)
        return init_val + frac * (final_val - init_val)

    def update_entropy_coef(self, it, total_it):
        """按迭代进度线性退火 entropy_coef：前 start_frac 段恒定初值，
        之后线性降到 final。精度任务需要前期探索、后期安静。
        同时退火底座残差的 entropy_coef（防止其 std 持续膨胀成噪声源）。"""
        if self.entropy_coef_final is not None:
            self.entropy_coef = self._anneal_linear(
                it, total_it, self.entropy_coef_init,
                self.entropy_coef_final, self.entropy_anneal_start_frac
            )
        if self.train_base and self.base_entropy_coef_final is not None:
            self.base_entropy_coef = self._anneal_linear(
                it, total_it, self.base_entropy_coef_init,
                self.base_entropy_coef_final, self.base_entropy_anneal_start_frac
            )

    def update_simple(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0
        mean_gate_loss = 0.0
        updates = 0
        ppo_grad_norm = 0.0
        payload_grad_norm = 0.0
        gate_grad_norm = 0.0
        value_loss_sums = torch.zeros(self.value_dim, device=self.device)
        return_sums = torch.zeros(self.value_dim, device=self.device)
        advantage_std_sums = torch.zeros(self.value_dim, device=self.device)
        payload_error_sums = {
            key: 0.0
            for key in ("total", "position", "torque", "velocity")
        }
        predicted_payload_sums = torch.zeros(9, device=self.device)
        true_payload_sums = torch.zeros(9, device=self.device)
        predicted_weight_sums = torch.zeros(3, device=self.device)
        predicted_weight_sq_sums = torch.zeros(3, device=self.device)
        prior_weight_sums = torch.zeros(3, device=self.device)
        actual_weight_sums = torch.zeros(3, device=self.device)
        actual_weight_sq_sums = torch.zeros(3, device=self.device)
        weight_entropy_sum = 0.0
        weight_change_sum = 0.0
        weight_change_count = 0
        weight_kl_sum = 0.0
        weight_smooth_sum = 0.0
        fused_return_sum = 0.0
        actor_entropy_sum = 0.0
        actor_kl_sum = 0.0
        actor_kl_count = 0
        std_clamp_fraction_sum = 0.0
        expert_gate_sums = None
        expert_gate_sq_sums = None
        expert_gate_entropy_sum = 0.0
        risk_sums = {
            key: 0.0
            for key in ("risk", "position", "torque", "velocity")
        }

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs
        )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            next_critic_obs_batch,
            target_values_batch,
            advantages_batch,
            fused_advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            load_weights_batch,
            previous_load_weights_batch,
            load_weight_smooth_mask_batch,
            payload_predictions_batch,
            base_batch,
            residual_context_batch,
        ) in generator:

            # 1) 用当前 policy 重新建立动作分布
            self.actor_critic.act(
                obs_batch,
                residual_context=residual_context_batch,
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            perception = getattr(self.actor_critic, "last_perception", None)
            if isinstance(perception, dict):
                expert_gate = perception.get("gate_prob")
                if expert_gate is not None:
                    if expert_gate_sums is None:
                        expert_gate_sums = torch.zeros(
                            expert_gate.size(-1), device=self.device
                        )
                        expert_gate_sq_sums = torch.zeros_like(
                            expert_gate_sums
                        )
                    expert_gate_sums += expert_gate.mean(dim=0).detach()
                    expert_gate_sq_sums += expert_gate.pow(2).mean(
                        dim=0
                    ).detach()
                    expert_gate_entropy_sum += weight_entropy(
                        expert_gate.detach()
                    ).mean().item()

            # 2) Actor distribution KL and optional adaptive learning rate.
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log((sigma_batch + 1e-8) / (old_sigma_batch + 1e-8))
                    + (
                        old_sigma_batch.pow(2)
                        + (old_mu_batch - mu_batch).pow(2)
                    ) / (2.0 * (sigma_batch.pow(2) + 1e-8))
                    - 0.5,
                    dim=-1
                )
                kl_mean = kl.mean()
                actor_kl_sum += kl_mean.item()
                actor_kl_count += 1

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        # 下限抬到 2e-4：std 退火后 σ 变小会让 KL 易爆，
                        # 原 1e-5 地板会导致后期 LR 触底、精度学不动
                        self.learning_rate = max(2e-4, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = (
                            self.learning_rate
                            * float(
                                param_group.get("lr_multiplier", 1.0)
                            )
                        )

            # 3) PPO surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            if self.value_dim > 1:
                # Rollout-time weights are fixed targets for this PPO update.
                # They are detached and therefore cannot be optimized through
                # the actor loss.
                adv = fused_advantages_batch.squeeze(-1).detach()
            else:
                adv = advantages_batch.squeeze(-1)

            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 4) Independent critic losses. Dynamic load weights affect only
            # advantage fusion, never the value-regression objective.
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                vl = torch.max(value_losses, value_losses_clipped)
            else:
                vl = (returns_batch - value_batch).pow(2)

            if self.value_dim > 1:
                value_loss_vector = vl.mean(dim=0)
                value_loss = value_loss_vector.mean()
            else:
                value_loss = vl.mean()
                value_loss_vector = value_loss.reshape(1)

            # 5) PPO loss
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # 6) optimize
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ppo_grad_norm += nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in self.optimizer.param_groups
                    for parameter in group["params"]
                ],
                self.max_grad_norm,
            ).item()
            self.optimizer.step()

            enforce_std_bounds = getattr(
                self.actor_critic, "enforce_std_bounds", None
            )
            if callable(enforce_std_bounds):
                clamp_fraction = enforce_std_bounds()
                std_clamp_fraction_sum += float(
                    clamp_fraction.item()
                )
            elif hasattr(self.actor_critic, "log_std"):
                with torch.no_grad():
                    self.actor_critic.log_std.clamp_(
                        math.log(1e-4), math.log(0.5)
                    )
            actor_entropy_sum += entropy_batch.mean().item()

            # 7) Payload supervision with a dedicated optimizer. The actor
            # consumes detached estimates, so this loss cannot update policy
            # or critic parameters.
            payload_prediction = None
            payload_target = None
            payload_loss = torch.zeros((), device=self.device)
            prop_encoder = getattr(self.actor_critic, "prop_encoder", None)
            payload_estimator = getattr(
                self.actor_critic, "payload_estimator", None
            )
            if (
                self.payload_slice is not None
                and self.payload_optimizer is not None
            ):
                if payload_estimator is not None:
                    payload_prediction = payload_estimator(obs_batch)
                elif hasattr(prop_encoder, "forward_full"):
                    _, _, payload_prediction = prop_encoder.forward_full(
                        obs_batch
                    )
                if payload_prediction is not None:
                    payload_target = critic_obs_batch[:, self.payload_slice].detach()
                    payload_errors = payload_prediction_errors(
                        payload_prediction, payload_target
                    )
                    payload_loss = (
                        self.payload_loss_coef * payload_errors["total"]
                    )
                    self.payload_optimizer.zero_grad(set_to_none=True)
                    payload_loss.backward()
                    payload_grad_norm += nn.utils.clip_grad_norm_(
                        self.payload_optimizer.param_groups[0]["params"],
                        self.max_grad_norm,
                    ).item()
                    self.payload_optimizer.step()
                    for key, value in payload_errors.items():
                        payload_error_sums[key] += value.item()
                    predicted_payload_sums += (
                        payload_prediction.detach().mean(dim=0)
                    )
                    true_payload_sums += payload_target.mean(dim=0)

            # 8) Load-weight supervision. Gate inputs and previous weights are
            # detached; only the gate MLP receives gradients.
            gate_loss = torch.zeros((), device=self.device)
            if (
                self.load_aware_enabled
                and self.load_gate_optimizer is not None
                and payload_prediction is not None
                and payload_target is not None
            ):
                predicted_weights, _, _ = self.actor_critic.load_weight_net(
                    payload_prediction.detach(), return_logits=True
                )
                prior = physical_prior_weights(
                    payload_target,
                    stability_min=self.load_stab_weight_min,
                    stability_max=self.load_stab_weight_max,
                    risk_threshold=self.load_risk_threshold,
                    risk_slope=self.load_risk_slope,
                    task_energy_ratio=self.task_energy_ratio,
                    position_coef=self.load_risk_position_coef,
                    torque_coef=self.load_risk_torque_coef,
                    velocity_coef=self.load_risk_velocity_coef,
                )
                prior_weights = prior["weights"].detach()
                supervision_loss = (
                    prior_weights
                    * (
                        prior_weights.clamp_min(1e-8).log()
                        - predicted_weights.clamp_min(1e-8).log()
                    )
                ).sum(dim=-1).mean()
                smooth_errors = (
                    predicted_weights - previous_load_weights_batch.detach()
                ).pow(2).sum(dim=-1, keepdim=True)
                smooth_mask = load_weight_smooth_mask_batch
                smooth_loss = (
                    (smooth_errors * smooth_mask).sum()
                    / smooth_mask.sum().clamp_min(1.0)
                )
                gate_loss = (
                    self.load_weight_supervision_coef * supervision_loss
                    + self.load_weight_smooth_coef * smooth_loss
                )
                self.load_gate_optimizer.zero_grad(set_to_none=True)
                gate_loss.backward()
                gate_grad_norm += nn.utils.clip_grad_norm_(
                    self.load_gate_optimizer.param_groups[0]["params"],
                    self.max_grad_norm,
                ).item()
                self.load_gate_optimizer.step()
                weight_change_sum += (
                    (smooth_errors.detach() * smooth_mask)
                    .sum()
                    .item()
                )
                weight_kl_sum += supervision_loss.item()
                weight_smooth_sum += smooth_loss.item()
                weight_change_count += int(smooth_mask.sum().item())
                predicted_weight_sums += predicted_weights.mean(dim=0).detach()
                predicted_weight_sq_sums += predicted_weights.pow(2).mean(
                    dim=0
                ).detach()
                prior_weight_sums += prior_weights.mean(dim=0)
                weight_entropy_sum += weight_entropy(
                    predicted_weights.detach()
                ).mean().item()
                for key in risk_sums:
                    risk_sums[key] += prior[key].mean().item()

            if (
                load_weights_batch is not None
                and load_weights_batch.abs().sum() > 0
            ):
                actual_weights = load_weights_batch
                actual_weight_sums += actual_weights.mean(dim=0)
                actual_weight_sq_sums += actual_weights.pow(2).mean(dim=0)
            if self.value_dim > 1 and actual_weights is not None:
                fused_return_sum += (
                    returns_batch * actual_weights
                ).sum(dim=-1).mean().item()
            elif self.value_dim > 1:
                actual_weights = self.value_weights.unsqueeze(0).expand(
                    actions_batch.size(0), -1
                )
                actual_weight_sums += actual_weights.mean(dim=0)
                actual_weight_sq_sums += actual_weights.pow(2).mean(dim=0)

            value_loss_sums += value_loss_vector.detach()
            return_sums += returns_batch.mean(dim=0).detach()
            advantage_std_sums += advantages_batch.std(dim=0).detach()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += payload_loss.item()
            mean_gate_loss += gate_loss.item()

            # 9) 底座顺应残差更新（共享融合 advantage，独立 optimizer）
            if self.train_base and base_batch[0] is not None:
                base_obs_batch, base_actions_batch, base_old_logp_batch, _, _ = base_batch

                self.base_residual.update_distribution(base_obs_batch)
                base_logp = self.base_residual.get_actions_log_prob(base_actions_batch)
                base_entropy = self.base_residual.entropy

                base_ratio = torch.exp(base_logp - base_old_logp_batch.squeeze(-1))
                base_surrogate = -adv * base_ratio
                base_surrogate_clipped = -adv * torch.clamp(
                    base_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                base_surrogate_loss = torch.max(base_surrogate, base_surrogate_clipped).mean()
                base_loss = base_surrogate_loss - self.base_entropy_coef * base_entropy.mean()

                self.base_optimizer.zero_grad()
                base_loss.backward()
                nn.utils.clip_grad_norm_(self.base_residual.parameters(), self.max_grad_norm)
                self.base_optimizer.step()

                with torch.no_grad():
                    self.base_residual.log_std.clamp_(math.log(1e-4), math.log(self.base_log_std_max))

            updates += 1

        updates = max(updates, 1)
        mean_value_loss /= updates
        mean_surrogate_loss /= updates
        mean_estimation_loss /= updates
        mean_gate_loss /= updates
        value_loss_sums /= updates
        return_sums /= updates
        advantage_std_sums /= updates
        for key in payload_error_sums:
            payload_error_sums[key] /= updates

        self.last_update_stats = {
            "surrogate_loss": mean_surrogate_loss,
            "value_loss": mean_value_loss,
            "payload_loss": mean_estimation_loss,
            "gate_loss": mean_gate_loss,
            "ppo_grad_norm": ppo_grad_norm / updates,
            "payload_grad_norm": payload_grad_norm / updates,
            "gate_grad_norm": gate_grad_norm / updates,
            "total_fused_return": fused_return_sum / updates,
            "actor_entropy": actor_entropy_sum / updates,
            "actor_kl": (
                actor_kl_sum / max(actor_kl_count, 1)
            ),
            "std_clamp_fraction": (
                std_clamp_fraction_sum / updates
            ),
        }
        raw_adv_std = getattr(
            self.storage, "raw_advantage_std", None
        )
        if raw_adv_std is not None:
            for index in range(self.value_dim):
                self.last_update_stats[
                    f"raw_advantage_std_{index}"
                ] = raw_adv_std[index].item()
        fused_adv_std = getattr(
            self.storage, "fused_advantage_std", None
        )
        if fused_adv_std is not None:
            self.last_update_stats["fused_advantage_std"] = (
                fused_adv_std.reshape(-1)[0].item()
            )
        component_rms = getattr(
            self.storage, "fused_advantage_component_rms", None
        )
        if component_rms is not None:
            for index in range(self.value_dim):
                self.last_update_stats[
                    f"fused_advantage_component_rms_{index}"
                ] = component_rms[index].item()
        for index in range(self.value_dim):
            self.last_update_stats[
                f"critic_value_loss_{index}"
            ] = value_loss_sums[index].item()
            self.last_update_stats[
                f"critic_return_{index}"
            ] = return_sums[index].item()
            self.last_update_stats[
                f"critic_advantage_std_{index}"
            ] = advantage_std_sums[index].item()
        for key, value in payload_error_sums.items():
            self.last_update_stats[f"payload_{key}_mse"] = value
        predicted_payload = predicted_payload_sums / updates
        true_payload = true_payload_sums / updates
        for index in range(9):
            self.last_update_stats[
                f"payload_predicted_{index}"
            ] = predicted_payload[index].item()
            self.last_update_stats[
                f"payload_true_{index}"
            ] = true_payload[index].item()
        self.last_update_stats["payload_predicted_com_norm"] = (
            predicted_payload[0:3].norm().item()
        )
        self.last_update_stats["payload_true_com_norm"] = (
            true_payload[0:3].norm().item()
        )
        self.last_update_stats["payload_predicted_torque_norm"] = (
            predicted_payload[3:6].norm().item()
        )
        self.last_update_stats["payload_true_torque_norm"] = (
            true_payload[3:6].norm().item()
        )
        self.last_update_stats["payload_predicted_velocity_norm"] = (
            predicted_payload[6:9].norm().item()
        )
        self.last_update_stats["payload_true_velocity_norm"] = (
            true_payload[6:9].norm().item()
        )
        if self.value_dim > 1 and not self.load_aware_enabled:
            actual_weight_mean = actual_weight_sums / updates
            actual_weight_std = (
                actual_weight_sq_sums / updates
                - actual_weight_mean.pow(2)
            ).clamp_min(0.0).sqrt()
            for index, name in enumerate(("task", "stability", "energy")):
                self.last_update_stats[
                    f"actual_{name}_weight"
                ] = actual_weight_mean[index].item()
                self.last_update_stats[
                    f"actual_{name}_weight_std"
                ] = actual_weight_std[index].item()
        if self.load_aware_enabled:
            predicted_weight_mean = predicted_weight_sums / updates
            predicted_weight_std = (
                predicted_weight_sq_sums / updates
                - predicted_weight_mean.pow(2)
            ).clamp_min(0.0).sqrt()
            prior_weight_mean = prior_weight_sums / updates
            actual_weight_mean = actual_weight_sums / updates
            actual_weight_std = (
                actual_weight_sq_sums / updates
                - actual_weight_mean.pow(2)
            ).clamp_min(0.0).sqrt()
            for index, name in enumerate(("task", "stability", "energy")):
                self.last_update_stats[
                    f"predicted_{name}_weight"
                ] = predicted_weight_mean[index].item()
                self.last_update_stats[
                    f"predicted_{name}_weight_std"
                ] = predicted_weight_std[index].item()
                self.last_update_stats[
                    f"prior_{name}_weight"
                ] = prior_weight_mean[index].item()
                self.last_update_stats[
                    f"actual_{name}_weight"
                ] = actual_weight_mean[index].item()
                self.last_update_stats[
                    f"actual_{name}_weight_std"
                ] = actual_weight_std[index].item()
            self.last_update_stats["weight_entropy"] = (
                weight_entropy_sum / updates
            )
            self.last_update_stats["weight_change_norm"] = (
                weight_change_sum / max(weight_change_count, 1)
            )
            self.last_update_stats["weight_kl"] = weight_kl_sum / updates
            self.last_update_stats["weight_smooth_loss"] = (
                weight_smooth_sum / updates
            )
            for key, value in risk_sums.items():
                self.last_update_stats[f"risk_{key}"] = value / updates
        if expert_gate_sums is not None:
            expert_gate_mean = expert_gate_sums / updates
            expert_gate_std = (
                expert_gate_sq_sums / updates - expert_gate_mean.pow(2)
            ).clamp_min(0.0).sqrt()
            for mode in range(expert_gate_mean.numel()):
                self.last_update_stats[
                    f"expert_gate_weight_{mode}"
                ] = expert_gate_mean[mode].item()
                self.last_update_stats[
                    f"expert_gate_weight_std_{mode}"
                ] = expert_gate_std[mode].item()
            self.last_update_stats["expert_gate_entropy"] = (
                expert_gate_entropy_sum / updates
            )
            uniform = torch.full_like(
                expert_gate_mean, 1.0 / expert_gate_mean.numel()
            )
            self.last_update_stats["expert_gate_load_balance"] = (
                expert_gate_mean - uniform
            ).abs().sum().item()

        self.storage.clear()
        if hasattr(self.actor_critic, "update_ema_target"):
            self.actor_critic.update_ema_target()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_estimation_loss,
            mean_gate_loss,
        )
