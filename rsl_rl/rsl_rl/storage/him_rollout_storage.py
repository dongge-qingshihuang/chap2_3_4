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

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class HIMRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.timeouts = None
            self.timeout_bootstrap_values = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.next_critic_observations = None
            self.payload_predictions = None
            self.load_weights = None
            # --- 底座顺应残差（联合 fine-tune 时使用，共享 critic/advantage）---
            self.base_observations = None
            self.base_actions = None
            self.base_actions_log_prob = None
            self.base_action_mean = None
            self.base_action_sigma = None
            self.residual_context = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu',
                 base_obs_shape=None, base_actions_shape=None,
                 residual_context_shape=None, value_dim=1):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        # 值函数维度：1 = 单评论家，>1 = 多评论家（各头独立 GAE）
        self.value_dim = value_dim

        # 是否额外存储底座残差的 transition（联合 fine-tune）
        self.store_base = base_obs_shape is not None and base_actions_shape is not None
        if self.store_base:
            self.base_observations = torch.zeros(num_transitions_per_env, num_envs, *base_obs_shape, device=self.device)
            self.base_actions = torch.zeros(num_transitions_per_env, num_envs, *base_actions_shape, device=self.device)
            self.base_actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.base_mu = torch.zeros(num_transitions_per_env, num_envs, *base_actions_shape, device=self.device)
            self.base_sigma = torch.zeros(num_transitions_per_env, num_envs, *base_actions_shape, device=self.device)

        self.store_residual_context = residual_context_shape is not None
        if self.store_residual_context:
            self.residual_context = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *residual_context_shape,
                device=self.device,
            )

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
            self.next_privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.timeouts = torch.zeros(
            num_transitions_per_env, num_envs, device=self.device
        ).bool()
        self.timeout_bootstrap_values = torch.zeros(
            num_transitions_per_env, num_envs, value_dim, device=self.device
        )

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=self.device)
        self.fused_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.raw_advantage_std = torch.zeros(value_dim, device=self.device)
        self.fused_advantage_std = torch.zeros(1, device=self.device)
        self.fused_advantage_component_rms = torch.zeros(
            value_dim, device=self.device
        )
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        if value_dim > 1:
            self.payload_predictions = torch.zeros(
                num_transitions_per_env, num_envs, 9, device=self.device
            )
            self.load_weights = torch.zeros(
                num_transitions_per_env, num_envs, value_dim, device=self.device
            )
            self.actual_load_weights = torch.zeros_like(self.load_weights)
            self.previous_load_weights = torch.zeros_like(self.load_weights)
            self.load_weight_smooth_mask = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
        else:
            self.payload_predictions = None
            self.load_weights = None
            self.actual_load_weights = None
            self.previous_load_weights = None
            self.load_weight_smooth_mask = None

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        if self.next_privileged_observations is not None: self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, self.value_dim))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        if transition.timeouts is not None:
            self.timeouts[self.step].copy_(
                transition.timeouts.reshape(-1)
            )
        if transition.timeout_bootstrap_values is not None:
            self.timeout_bootstrap_values[self.step].copy_(
                transition.timeout_bootstrap_values
            )
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if self.load_weights is not None:
            if transition.load_weights is not None:
                self.load_weights[self.step].copy_(transition.load_weights)
                if self.step == 0:
                    self.previous_load_weights[self.step].copy_(
                        transition.load_weights
                    )
                    self.load_weight_smooth_mask[self.step].zero_()
                else:
                    self.previous_load_weights[self.step].copy_(
                        self.load_weights[self.step - 1]
                    )
                    self.load_weight_smooth_mask[self.step].copy_(
                        1.0 - self.dones[self.step - 1].float()
                    )
            if transition.payload_predictions is not None:
                self.payload_predictions[self.step].copy_(
                    transition.payload_predictions
                )
        if self.store_base:
            self.base_observations[self.step].copy_(transition.base_observations)
            self.base_actions[self.step].copy_(transition.base_actions)
            self.base_actions_log_prob[self.step].copy_(transition.base_actions_log_prob.view(-1, 1))
            self.base_mu[self.step].copy_(transition.base_action_mean)
            self.base_sigma[self.step].copy_(transition.base_action_sigma)
        if self.store_residual_context:
            if transition.residual_context is None:
                raise RuntimeError(
                    "Residual context is required by rollout storage"
                )
            self.residual_context[self.step].copy_(
                transition.residual_context
            )
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            timeout = self.timeouts[step].unsqueeze(-1)
            next_values = torch.where(
                timeout,
                self.timeout_bootstrap_values[step],
                next_values,
            )
            not_done = 1.0 - self.dones[step].float()
            # Timeout bootstraps the value of the terminal observation, but it
            # still ends the episode. Never recurse GAE into the reset episode.
            bootstrap = (
                not_done.bool() | timeout
            ).float()
            advantage_continues = (
                not_done * (1.0 - timeout.float())
            )
            delta = (
                self.rewards[step]
                + bootstrap * gamma * next_values
                - self.values[step]
            )
            advantage = (
                delta
                + advantage_continues * gamma * lam * advantage
            )
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages.
        # 逐头在整个 rollout 上分别标准化：多评论家下各组奖励量级差几十倍也不影响，
        # λ 因此只表达相对重要性。value_dim=1 时与原来的全局归一化完全等价。
        self.advantages = self.returns - self.values
        self.raw_advantage_std.copy_(
            self.advantages.std(dim=(0, 1))
        )
        mean = self.advantages.mean(dim=(0, 1), keepdim=True)
        std = self.advantages.std(dim=(0, 1), keepdim=True)
        self.advantages = (self.advantages - mean) / (std + 1e-8)

    def compute_fused_advantages(self, fallback_weights=None):
        """Fuse normalized critic advantages with rollout-time weights."""
        if self.value_dim == 1:
            self.fused_advantages = self.advantages
            self.fused_advantage_std.copy_(
                self.fused_advantages.std(unbiased=False).reshape(1)
            )
            self.fused_advantage_component_rms.copy_(
                self.fused_advantages.pow(2).mean().sqrt().reshape(1)
            )
            return self.fused_advantages
        if self.load_weights is not None and self.load_weights.abs().sum() > 0:
            weights = self.load_weights
        elif fallback_weights is not None:
            weights = fallback_weights.to(self.device).view(1, 1, -1)
            weights = weights.expand(
                self.num_transitions_per_env, self.num_envs, -1
            )
        else:
            raise RuntimeError(
                "Multi-critic advantage fusion requires load weights or "
                "fallback critic weights"
            )
        self.actual_load_weights.copy_(weights)
        weighted_components = self.advantages * weights
        self.fused_advantage_component_rms.copy_(
            weighted_components.pow(2).mean(dim=(0, 1)).sqrt()
        )
        self.fused_advantages = weighted_components.sum(
            dim=-1, keepdim=True
        )
        self.fused_advantage_std.copy_(
            self.fused_advantages.std(unbiased=False).reshape(1)
        )
        return self.fused_advantages

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
            next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        fused_advantages = self.fused_advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        if self.load_weights is not None:
            load_weights = self.actual_load_weights.flatten(0, 1)
            previous_load_weights = self.previous_load_weights.flatten(0, 1)
            load_weight_smooth_mask = self.load_weight_smooth_mask.flatten(0, 1)
            payload_predictions = self.payload_predictions.flatten(0, 1)
        else:
            load_weights = None
            previous_load_weights = None
            load_weight_smooth_mask = None
            payload_predictions = None

        if self.store_base:
            base_observations = self.base_observations.flatten(0, 1)
            base_actions = self.base_actions.flatten(0, 1)
            base_old_log_prob = self.base_actions_log_prob.flatten(0, 1)
            base_old_mu = self.base_mu.flatten(0, 1)
            base_old_sigma = self.base_sigma.flatten(0, 1)

        if self.store_residual_context:
            residual_context = self.residual_context.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_critic_observations_batch = next_critic_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                fused_advantages_batch = fused_advantages[batch_idx]
                if load_weights is not None:
                    load_weights_batch = load_weights[batch_idx]
                    previous_load_weights_batch = previous_load_weights[batch_idx]
                    load_weight_smooth_mask_batch = load_weight_smooth_mask[batch_idx]
                    payload_predictions_batch = payload_predictions[batch_idx]
                else:
                    load_weights_batch = None
                    previous_load_weights_batch = None
                    load_weight_smooth_mask_batch = None
                    payload_predictions_batch = None

                if self.store_base:
                    base_batch = (
                        base_observations[batch_idx],
                        base_actions[batch_idx],
                        base_old_log_prob[batch_idx],
                        base_old_mu[batch_idx],
                        base_old_sigma[batch_idx],
                    )
                else:
                    base_batch = (None, None, None, None, None)

                residual_context_batch = (
                    residual_context[batch_idx]
                    if self.store_residual_context
                    else None
                )

                yield (
                    obs_batch,
                    critic_observations_batch,
                    actions_batch,
                    next_critic_observations_batch,
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
                )
