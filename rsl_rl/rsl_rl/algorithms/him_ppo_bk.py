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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage
from rsl_rl.modules import ActorCritic_dream
# from rsl_rl.modules import ActorCritic_woattention
from rsl_rl.modules import ActorCritic_attention
from rsl_rl.modules import ActorCritic_attention1

class HIMPPO:
    actor_critic: ActorCritic_attention1
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
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

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
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

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
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                estimation_loss, swap_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch, lr=self.learning_rate)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss


    def update_dream(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0   # 这里用作 vel_loss mean
        mean_swap_loss = 0.0         # 这里用作 recon_loss mean

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 一些权重（如果你外面没定义，就用默认值）
        beta_kl = getattr(self, "beta_kl", 1.0)
        vae_loss_coef = getattr(self, "vae_loss_coef", 1.0)
        vel_w = getattr(self, "vel_w", 1.0)
        recon_w = getattr(self, "recon_w", 1.0)

        for (obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch,
            target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch,
            old_mu_batch, old_sigma_batch) in generator:

            # -------------------------
            # 1) Policy forward (act 只为更新分布)
            # -------------------------
            # 如果你的 ActorCritic.act 支持 obs_hist，可用 kwargs 传入；不支持也不会报错（因为原来有 **kwargs）
            try:
                self.actor_critic.act(obs_batch, obs_hist=obs_batch)
            except TypeError:
                self.actor_critic.act(obs_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # -------------------------
            # 2) Adaptive KL（保留）
            # -------------------------
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1e-8) + 1e-5)
                        + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
                        / (2.0 * (sigma_batch.pow(2) + 1e-8))
                        - 0.5,
                        dim=-1
                    )
                    kl_mean = kl.mean()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # -------------------------
            # 3) VAE loss: vel supervised + recon(next_obs) + beta*KL
            # -------------------------
            # target: next obs（默认取 next_critic_obs 的前 obs_dim 维）
            obs_next_target = next_critic_obs_batch[:, :45].detach()

            # target: next velocity（你之前用 45:48，沿用这个约定）
            vel_target = critic_obs_batch[:, 45:48].detach()

            # forward VAE（兼容两种命名：vae_forward 或 cenet_forward）
            if hasattr(self.actor_critic, "vae_forward"):
                vel_pred, z, obs_next_pred, mu_latent, logvar_latent = self.actor_critic.vae_forward(obs_batch)
            else:
                raise AttributeError("actor_critic must provide vae_forward(obs_hist) or cenet_forward(obs_hist)")

            vel_loss = nn.MSELoss()(vel_pred, vel_target)
            recon_loss = nn.MSELoss()(obs_next_pred, obs_next_target)

            # KL(q(z|x)||N(0,1))：用 mean（别用 sum）
            kl_loss = -0.5 * torch.mean(1.0 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp())

            vae_loss = vel_w * vel_loss + recon_w * recon_loss + beta_kl * kl_loss

            # -------------------------
            # 4) PPO surrogate + value
            # -------------------------
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            adv = advantages_batch.squeeze(-1)

            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总 loss：PPO + VAE
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + vae_loss_coef * vae_loss
            )

            # -------------------------
            # 5) Backprop
            # -------------------------
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += vel_loss.item()     # 第3返回：速度监督损失
            mean_swap_loss += recon_loss.item()         # 第4返回：重构损失（如果想换成 vae_loss/kl_loss 改这里）

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates

        self.storage.clear()

        # 保持返回接口不变：4个值
        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss


    def update_1(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0   # vel supervised loss mean
        mean_swap_loss = 0.0         # 占位（保持返回接口），这里不再用 recon/KL

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        vel_loss_coef = getattr(self, "vel_loss_coef", 1.0)
        mse = nn.MSELoss()

        for (obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch,
            target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch,
            old_mu_batch, old_sigma_batch) in generator:

            # -------------------------
            # 1) Policy forward：保持你现在的逻辑不动
            # -------------------------
            
            self.actor_critic.act(obs_batch, obs_hist=obs_batch)


            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # -------------------------
            # 2) Adaptive KL（保留）
            # -------------------------
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1e-8) + 1.e-5)
                        + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
                        / (2.0 * (sigma_batch.pow(2) + 1e-8))
                        - 0.5,
                        dim=-1
                    )
                    kl_mean = kl.mean()

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # -------------------------
            # 3) 速度头监督学习（只要 vel_loss）
            #    - 速度目标：建议用 next_critic_obs_batch（预测下一时刻速度）
            # -------------------------
            vel_target = critic_obs_batch[:, 45:48].detach()

            if hasattr(self.actor_critic, "prop_encoder"):
                # prop_encoder 返回 (z_prop, vel_pred)
                _, vel_pred = self.actor_critic.prop_encoder(obs_batch)
            else:
                raise AttributeError("actor_critic must have prop_encoder(...) or vae_forward(...) to get vel_pred")

            vel_loss = mse(vel_pred, vel_target)

            # -------------------------
            # 4) PPO surrogate + value
            # -------------------------
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            adv = advantages_batch.squeeze(-1)

            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总 loss：PPO + 速度监督（不再包含 recon/KL）
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + vel_loss_coef * vel_loss
            )

            # -------------------------
            # 5) Backprop
            # -------------------------
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += vel_loss.item()
            # mean_swap_loss 维持为 0.0（占位）

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates

        self.storage.clear()

        # 返回接口不变：4个值
        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss



    def update_attention(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_pre_vel_loss = 0.0
        mean_swap_loss = 0.0  # 这里用来占位/返回：我先放 vae_loss(或 recon_loss)，你自己决定

        # 你原来就是这个 generator；但注意：下面解包按 update_attention 的字段来写
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # 地形分类 loss 系数（从 actor_critic 拿，或者默认 0.5）
        lambda_terrain_cls = getattr(self.actor_critic, "terrain_cls_coef", 0.5)

        for (obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch,
            target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch,
            old_mu_batch, old_sigma_batch) in generator:

            original_batch_size = obs_batch.shape[0]
            device = obs_batch.device

            # ==========================================================
            # 1) PPO policy forward：触发 TsDyn+注意力路由，更新 last_attn_weights
            # ==========================================================
            # 如果你 actor_critic.act 不用 masks/hidden，传了也会被 **kwargs 吃掉
            self.actor_critic.act(obs_batch, obs_hist=obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            # ==========================================================
            # 2) 速度预测监督（prop_encoder 输出 vel_pred）
            # ==========================================================
            # prop_encoder 返回 (z_prop, pre_vel)
            _, pre_vel = self.actor_critic.prop_encoder(obs_batch)

            # 这里目标你自己填：原版用 obs_future_batch[:, :3]
            # real_vel = obs_future_batch[:, :3]
            # 先放一个占位（避免你忘了）
            real_vel = critic_obs_batch[:, 45:48]  # TODO: 确认 obs_future_batch 的前三维确实是未来真实速度

            # 原版用 ramp + mix_velocity；你自己填函数/策略
            # p = linear_ramp(step=iter, start_step=1000, end_step=5000)
            # vel_in = mix_velocity(real_vel, pre_vel, prob_pred=p, mode="blend", detach_pred=False)
            vel_in = pre_vel  # TODO: 你要用 mix_velocity 就把这里换掉

            pre_vel_loss = F.mse_loss(pre_vel, real_vel).mean()

            # ==========================================================
            # 3) 注意力路由权重（来自 actor_critic.last_attn_weights）
            # ==========================================================
            attn_weights = getattr(self.actor_critic, "last_attn_weights", None)  # [B,K]
            if attn_weights is None:
                B = obs_batch.size(0)
                K = self.actor_critic.num_modes
                attn_weights = torch.full((B, K), 1.0 / K, device=device)

            route = attn_weights  # [B,K]  # 注意：这里不 detach，允许监督 loss 训练路由
            router_entropy = -(route * torch.log(route + 1e-8)).sum(dim=-1).mean()

            # ==========================================================
            # 4) 特权地形编码 + 重建 + EMA-KMeans 聚类得到伪标签 idx
            # ==========================================================
            H = self.actor_critic.terrain_dim
            heights = critic_obs_batch[:, 45+3+3:45+3+3+H].float()

            z_ter, heights_hat = self.actor_critic.terrain_encoder(heights, return_recon=True)
            idx, one_hot, pull_loss = self.actor_critic.terrain_clusterer(z_ter)
            idx = idx.long()

            ter_recon_loss = F.mse_loss(heights_hat, heights)

            pk = one_hot.float().mean(0)
            uniform = torch.full_like(pk, 1.0 / pk.numel())
            balance_loss = F.kl_div((pk + 1e-8).log(), uniform, reduction="batchmean")

            # ==========================================================
            # 5) TsDyn：每个 mode forward，算 recon/contact/terrainCE/KL
            # ==========================================================
            recon_loss_list = []
            recon_contact_loss_list = []
            terrain_ce_list = []
            mu_list, logvar_list = [], []

            for sub_net in self.actor_critic.TsDyn_modules:
                # VAE.forward 返回：recon_x, recon_contact, ter_logits, mu, logvar
                recon_x, recon_contact, ter_logits, mu, logvar = sub_net.forward(obs_batch)
                mu_list.append(mu)
                logvar_list.append(logvar)

                # 目标：未来 obs 去掉前 3 维 vel（你原版就是这样）
                rec_i = F.mse_loss(
                    recon_x,
                    next_critic_obs_batch[:, :45],  # TODO: 确认 future 的维度切分
                    reduction="none"
                ).sum(dim=1)
                recon_loss_list.append(rec_i)

                con_i = F.binary_cross_entropy(
                    recon_contact,
                    next_critic_obs_batch[:,-4:].float(),      # TODO: 确认 shape 对齐 [B, feet_contact_dim]
                    reduction="none"
                ).sum(dim=1)
                recon_contact_loss_list.append(con_i)

                ce_i = F.cross_entropy(
                    ter_logits,
                    idx,
                    reduction="none"
                )
                terrain_ce_list.append(ce_i)

            recon_loss_all = torch.stack(recon_loss_list, dim=1)                 # [B,K]
            recon_contact_loss_all = torch.stack(recon_contact_loss_list, dim=1) # [B,K]
            terrain_ce_all = torch.stack(terrain_ce_list, dim=1)                 # [B,K]

            recon_loss = (recon_loss_all * route).sum(dim=1).mean()
            recon_contact_loss = (recon_contact_loss_all * route).sum(dim=1).mean()
            terrain_cls_loss = (terrain_ce_all * route).sum(dim=1).mean()

            mu_stack = torch.stack(mu_list, dim=1)         # [B,K,Dz]
            logvar_stack = torch.stack(logvar_list, dim=1) # [B,K,Dz]
            kl_per_mode = -0.5 * (1 + logvar_stack - mu_stack.pow(2) - logvar_stack.exp()).sum(dim=-1)  # [B,K]
            kl_div = (kl_per_mode * route).sum(dim=1).mean()

            vae_loss = recon_loss + 0.8 * recon_contact_loss + 0.9 * kl_div

            coef_tr = getattr(self.actor_critic, "terrain_recon_coef", 1.0)
            coef_pl = getattr(self.actor_critic, "cluster_pull_coef", 0.25)
            coef_bl = getattr(self.actor_critic, "cluster_balance_coef", 0.10)
            terrain_aux = coef_tr * ter_recon_loss + coef_pl * pull_loss + coef_bl * balance_loss

            # ==========================================================
            # 6) PPO value & entropy
            # ==========================================================
            value_batch = self.actor_critic.evaluate(critic_obs_batch)

            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # ---- Adaptive KL（原样保留）----
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1.0e-5) + 1.0e-5)
                        + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2)) / (2.0 * sigma_batch.pow(2))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # ==========================================================
            # 7) PPO surrogate + value loss
            # ==========================================================
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ==========================================================
            # 8) 总损失（原 update_attention 的组合）
            # ==========================================================
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.entropy_coef * router_entropy
                + vae_loss
                + pre_vel_loss
                + terrain_aux
                + lambda_terrain_cls * terrain_cls_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # ==========================================================
            # 9) smoothness（你原来那套，如果你这版没有相关变量就先注释掉自己补）
            # ==========================================================
            # epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
            # policy_smooth_coef = self.smoothness_upper_bound * epsilon
            # value_smooth_coef = self.value_smoothness_coef * policy_smooth_coef
            #
            # mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
            # mix_critic_batch = critic_obs_batch + mix_weights * (next_critic_obs_batch - critic_obs_batch)
            # value_smooth_loss = torch.square(torch.norm(value_batch - self.actor_critic.evaluate(mix_critic_batch), dim=-1)).mean()
            # smooth_loss = value_smooth_coef * value_smooth_loss
            # loss += smooth_loss

            # ==========================================================
            # 10) backward
            # ==========================================================
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # stats（按你 update_1 的 4 返回值习惯先放这些）
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_pre_vel_loss += pre_vel_loss.item()
            mean_swap_loss += vae_loss.item()  # 你要换成 recon_loss/terrain_aux 都行

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_pre_vel_loss /= num_updates
        mean_swap_loss /= num_updates

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_pre_vel_loss, mean_swap_loss
