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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class ActorCriticArmServo(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.5,
        history_length=6,
        one_step_obs_dim=None,
        dog_obs_dim=45,
        arm_state_dim=18,   # arm_q 6 + arm_dq 6 + last_action 6
        cmd_dim=6,
        error_dim=6,       # pos_err 3 + ori_err 3 + d_pos_err 3 + d_ori_err 3
        **kwargs
    ):
        super().__init__()

        act = get_activation(activation)

        self.num_actions = num_actions
        self.history_length = history_length
        self.one_step_obs_dim = one_step_obs_dim or (num_actor_obs // history_length)

        self.dog_obs_dim = dog_obs_dim
        self.arm_state_dim = arm_state_dim
        self.cmd_dim = cmd_dim
        self.error_dim = error_dim

        # 按你的 one-step obs 顺序自己确认一下
        # 假设 one-step:
        # [dog 45, arm_q 6, arm_dq 6, cmd 6, last_action 6, error 6]
        self.dog_slice = slice(0, dog_obs_dim)
        self.arm_state_slice = slice(dog_obs_dim, dog_obs_dim + arm_state_dim)
        self.cmd_slice = slice(dog_obs_dim + arm_state_dim,
                               dog_obs_dim + arm_state_dim + cmd_dim)
        self.error_slice = slice(dog_obs_dim + arm_state_dim + cmd_dim,
                                 dog_obs_dim + arm_state_dim + cmd_dim + error_dim)

        dog_in = dog_obs_dim * history_length
        arm_in = arm_state_dim * history_length
        cmd_in = cmd_dim * history_length
        err_in = error_dim * history_length

        self.dog_encoder = nn.Sequential(
            nn.Linear(dog_in, 128), act,
            nn.Linear(128, 64), act,
        )

        self.arm_encoder = nn.Sequential(
            nn.Linear(arm_in + cmd_in, 128), act,
            nn.Linear(128, 64), act,
        )

        self.error_encoder = nn.Sequential(
            nn.Linear(err_in, 128), act,
            nn.Linear(128, 64), act,
        )

        self.actor_head = nn.Sequential(
            nn.Linear(64 + 64 + 64, actor_hidden_dims[0]), act,
            nn.Linear(actor_hidden_dims[0], actor_hidden_dims[1]), act,
            nn.Linear(actor_hidden_dims[1], num_actions),
        )

        # 误差反馈残差头
        self.servo_head = nn.Sequential(
            nn.Linear(error_dim, 64), act,
            nn.Linear(64, num_actions),
        )

        # 建议 servo 初始很小，避免一开始乱甩
        nn.init.zeros_(self.servo_head[-1].weight)
        nn.init.zeros_(self.servo_head[-1].bias)

        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(act)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(act)
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def _split_obs(self, obs):
        B = obs.shape[0]
        obs_h = obs.view(B, self.history_length, self.one_step_obs_dim)

        dog = obs_h[:, :, self.dog_slice].reshape(B, -1)
        arm = obs_h[:, :, self.arm_state_slice].reshape(B, -1)
        cmd = obs_h[:, :, self.cmd_slice].reshape(B, -1)
        err = obs_h[:, :, self.error_slice].reshape(B, -1)

        # 当前时刻 error，给 servo head 用
        err_now = obs_h[:, -1, self.error_slice]

        return dog, arm, cmd, err, err_now

    def update_distribution(self, observations):
        dog, arm, cmd, err, err_now = self._split_obs(observations)

        z_dog = self.dog_encoder(dog)
        z_arm = self.arm_encoder(torch.cat([arm, cmd], dim=-1))
        z_err = self.error_encoder(err)

        residual_action = self.actor_head(torch.cat([z_dog, z_arm, z_err], dim=-1))
        servo_action = self.servo_head(err_now)

        mean = residual_action + servo_action
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        dog, arm, cmd, err, err_now = self._split_obs(observations)

        z_dog = self.dog_encoder(dog)
        z_arm = self.arm_encoder(torch.cat([arm, cmd], dim=-1))
        z_err = self.error_encoder(err)

        residual_action = self.actor_head(torch.cat([z_dog, z_arm, z_err], dim=-1))
        servo_action = self.servo_head(err_now)

        return residual_action + servo_action

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def reset(self, dones=None):
        pass

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
