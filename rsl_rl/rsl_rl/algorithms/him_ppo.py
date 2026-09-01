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
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
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
        else:
            self.value_weights = None

        # --- 第二章负载感知：黑盒估计器的监督（特权真值 -> ProprioEncoder 预测）---
        # 由 runner 从 env 写入；None 表示未启用
        self.payload_slice = None
        self.payload_loss_coef = 1.0

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
                     base_obs_shape=None, base_actions_shape=None):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device,
                                         base_obs_shape=base_obs_shape, base_actions_shape=base_actions_shape,
                                         value_dim=self.value_dim)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
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
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            # values 是 (N, value_dim)，rewards 单评论家下是 (N,)、多评论家下是 (N, value_dim)，
            # 统一按 rewards 的形状对齐后再累加
            bootstrap = self.gamma * self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device)
            self.transition.rewards += bootstrap.reshape(self.transition.rewards.shape)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

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
        mean_estimation_loss = 0.0   # simple ActorCritic 没 estimator，这里占位返回 0
        mean_swap_loss = 0.0         # simple ActorCritic 没 swap loss，这里占位返回 0

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
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            base_batch
        ) in generator:

            # 1) 用当前 policy 重新建立动作分布
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # 2) adaptive KL（保留你原来的逻辑）
            if self.desired_kl is not None and self.schedule == "adaptive":
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

                    if kl_mean > self.desired_kl * 2.0:
                        # 下限抬到 2e-4：std 退火后 σ 变小会让 KL 易爆，
                        # 原 1e-5 地板会导致后期 LR 触底、精度学不动
                        self.learning_rate = max(2e-4, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 3) PPO surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            if self.value_dim > 1:
                # 各头 advantage 已在 storage 中分别标准化，这里按 λ 加权成标量。
                # 末尾再归一化一次，使结果只取决于 λ 的比例、与其整体缩放无关，
                # advantage 尺度也和单评论家一致（clip / 学习率行为可比）
                adv = (advantages_batch * self.value_weights.unsqueeze(0)).sum(dim=-1)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            else:
                adv = advantages_batch.squeeze(-1)

            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 4) value loss（多评论家时各头分别算，再按同一组 λ 加权求和）
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
                value_loss = (vl * self.value_weights.unsqueeze(0)).sum(dim=-1).mean()
            else:
                value_loss = vl.mean()

            # 5) total loss
            aux_loss = torch.zeros((), device=self.device)
            arm_delta_loss = torch.zeros((), device=self.device)
            zero_loss = torch.zeros((), device=self.device)

            # aux_loss, arm_delta_loss, zero_loss = self.actor_critic.auxiliary_loss(
            #     obs_batch,
            #     actions_batch,
            #     critic_obs_batch,
            #     next_critic_obs_batch,
            # )

            # 第二章：负载状态估计的监督损失。ProprioEncoder 只看本体观测，
            # 用特权通道里的解析真值做回归，使部署时无需臂编码器即可获得负载感知
            payload_loss = torch.zeros((), device=self.device)
            prop_encoder = getattr(self.actor_critic, "prop_encoder", None)
            if self.payload_slice is not None and hasattr(prop_encoder, "forward_full"):
                _, _, payload_pred = prop_encoder.forward_full(obs_batch)
                if payload_pred is not None:
                    payload_target = critic_obs_batch[:, self.payload_slice].detach()
                    payload_loss = F.mse_loss(payload_pred, payload_target)

            aux_loss = 0
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.aux_loss_coef * aux_loss
                + self.payload_loss_coef * payload_loss
            )

            # 6) optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # log_std 参数投影：把参数本身夹回 [log(1e-4), log(0.5)]。
            # 若只在 std 属性里 clamp 输出，log_std 一旦越过上限梯度即为零，
            # std 会永久钉死在上限（吸收态），熵系数调多小都无法恢复。
            # 区间需与 actor_critic 中 std 属性的 clamp 范围保持一致。
            if hasattr(self.actor_critic, "log_std"):
                with torch.no_grad():
                    self.actor_critic.log_std.clamp_(math.log(1e-4), math.log(0.5))

            # 7) 底座顺应残差更新（共享 advantage，独立 optimizer + 低 LR）
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

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            # mean_estimation_loss += aux_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss
