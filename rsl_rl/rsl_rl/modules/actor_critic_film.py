import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Invalid activation: {act_name}")


def build_mlp(in_dim, hidden_dims, out_dim, activation="elu", layer_norm=True):
    act = get_activation(activation)
    layers = []
    last_dim = in_dim

    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act)
        last_dim = h

    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim, hidden_dim=128, activation="elu"):
        super().__init__()

        self.net = build_mlp(
            cond_dim,
            [hidden_dim],
            2 * feat_dim,
            activation=activation,
            layer_norm=False,
        )

        # 初始时 gamma=0, beta=0，让 FiLM 不破坏主策略
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)

    def forward(self, feat, cond):
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        # 约束调制幅度在 ±50%，防止 dog_z 分布漂移时特征尺度爆炸
        # tanh(0)=0，零初始化下仍是恒等映射，不改变初始行为
        gamma = 0.5 * torch.tanh(gamma)
        return feat * (1.0 + gamma) + beta


class ActorCriticArmServoDogPrior(nn.Module):
    """
    A-D 简化版：
    A. 去掉 TCN/GRU，换成轻量 ori_fast_encoder
    B. 关闭 auxiliary loss
    C. ori_action_head 零初始化，初始不扰动主策略
    D. 默认 init_noise_std 从 0.5 降到 0.2
    E. pos_encoder 加入 pos_err_delta（与 ori 通路对称的 P+D 快反馈）
    F. arm_err_hist_encoder：编码全部历史帧的 pose error 演化轨迹
    G. FiLM gamma 用 0.5*tanh 约束，防止特征尺度漂移

    Observation one-step layout:
        dog_obs: 45
            commands              0:3
            base_ang_vel          3:6
            projected_gravity     6:9
            dog_pos               9:21
            dog_vel               21:33
            dog_actions           33:45

        arm_obs: 30
            arm_pos               0:6
            arm_vel               6:12
            arm_cmd               12:18
            arm_pose_error        18:24
                pos_error         18:21
                ori_error         21:24
            prev_action           24:30
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_one_step_obs=75,
        history_length=None,
        activation="elu",
        init_noise_std=0.2,   # D: 姿态精度任务建议降低探索噪声
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        dog_feature_dim=128,
        dog_latent_dim=32,
        dog_prior_scale=0.1,
        **kwargs,
    ):
        super().__init__()

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs

        if history_length is None:
            assert num_actor_obs % num_one_step_obs == 0
            history_length = num_actor_obs // num_one_step_obs
        self.history_length = history_length

        self.dog_dim = 45
        self.arm_dim = self.num_one_step_obs - self.dog_dim
        # 第四章在 arm 块尾部追加 IK 名义解偏差；第三章下 ik_dim=0，布局与原来完全一致
        self.ik_dim = self.arm_dim - 30
        assert self.ik_dim >= 0, f"one-step obs 至少需含 45 dog + 30 arm，实际 {self.num_one_step_obs}"

        self.dog_prior_scale = dog_prior_scale
        self.actor_feat_dim = actor_hidden_dims[-1]

        # position error encoder: pos_err_now(3) + pos_err_delta_now(3) -> pos_z(64)
        # 与 ori 通路对称：误差 + 误差变化率（P + D）
        self.pos_encoder = build_mlp(
            6,
            [64, 64],
            64,
            activation=activation,
            layer_norm=False,
        )

        # A: lightweight orientation fast feedback encoder
        # input:
        #   rot_err_now          3
        #   rot_err_delta_now    3
        #   target_rpy           3
        #   arm_vel              6
        #   prev_action          6
        #   base_ang_vel         3
        #   projected_gravity    3
        # total = 27
        self.ori_fast_encoder = build_mlp(
            27,
            [128, 64],
            64,
            activation=activation,
            layer_norm=False,
        )

        # arm state = arm_pos 6 + arm_vel 6 + previous_action 6 (+ 第四章 IK 名义解偏差 6)
        self.arm_state_encoder = build_mlp(
            18 + self.ik_dim,
            [128, 128],
            128,
            activation=activation,
            layer_norm=False,
        )

        # command = desired l,p,y,roll,pitch,yaw
        self.arm_cmd_encoder = build_mlp(
            6,
            [64],
            32,
            activation=activation,
            layer_norm=False,
        )

        # dog/base history encoder
        self.dog_prior = build_mlp(
            self.dog_dim * self.history_length,
            [dog_feature_dim, dog_feature_dim],
            dog_latent_dim,
            activation=activation,
            layer_norm=False,
        )

        # arm pose error history encoder:
        # (pos_err 3 + ori_err 3) x history_length -> hist_z(32)
        # 只取误差通道的历史，编码误差在底座扰动下的演化轨迹
        self.arm_err_hist_encoder = build_mlp(
            6 * self.history_length,
            [64],
            32,
            activation=activation,
            layer_norm=False,
        )

        self.pre_fusion = build_mlp(
            128 + 64 + 64 + 32 + 32,  # arm_z + pos_z + ori_z + cmd_z + hist_z
            [256],
            self.actor_feat_dim,
            activation=activation,
            layer_norm=False,
        )

        self.base_film = FiLM(
            cond_dim=dog_latent_dim,
            feat_dim=self.actor_feat_dim,
            hidden_dim=128,
            activation=activation,
        )

        self.main_action_head = nn.Linear(self.actor_feat_dim, num_actions)

        self.dog_action_bias = build_mlp(
            dog_latent_dim,
            [64],
            num_actions,
            activation=activation,
            layer_norm=False,
        )

        # orientation residual action head
        self.ori_action_scale = kwargs.get("ori_action_scale", 0.5)
        # self.ori_action_scale = 0
        self.ori_action_head = build_mlp(
            64 + self.actor_feat_dim + dog_latent_dim,
            [128, 64],
            num_actions,
            activation=activation,
            layer_norm=False,
        )

        self.critic = build_mlp(
            num_critic_obs,
            list(critic_hidden_dims),
            1,
            activation=activation,
            layer_norm=False,
        )

        # 保留原来的 dog auxiliary heads，避免外部代码如果读取属性时报错
        self.dog_base_vel_head = build_mlp(
            dog_latent_dim,
            [64],
            3,
            activation=activation,
            layer_norm=False,
        )

        self.dog_disturbance_head = build_mlp(
            dog_latent_dim,
            [64],
            3,
            activation=activation,
            layer_norm=False,
        )

        self.log_std = nn.Parameter(
            torch.log(torch.ones(num_actions) * init_noise_std)
        )

        self.distribution = None
        self._init_weights()

    def _init_weights(self):
        # 主动作 head 小初始化
        nn.init.orthogonal_(self.main_action_head.weight, gain=0.01)
        nn.init.constant_(self.main_action_head.bias, 0.0)

        # dog prior 初始不影响动作
        last = self.dog_action_bias[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)

        # C: 姿态 residual head 零初始化，初始完全不影响主策略
        last = self.ori_action_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)

    @property
    def std(self):
        # 上限 0.5：delta-q 控制下更大的 std 会让末端抖动超过 settle 容差
        return torch.exp(self.log_std).clamp(1e-4, 0.5)

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

    def _split_obs(self, observations):
        b = observations.shape[0]
        x = observations.view(b, self.history_length, self.num_one_step_obs)
        dog = x[:, :, :self.dog_dim]
        return dog

    def split_actor_obs(self, observations):
        b = observations.shape[0]
        x = observations.view(b, self.history_length, self.num_one_step_obs)

        # dog history for dog latent
        dog_hist = x[:, :, :self.dog_dim]
        dog_input = dog_hist.reshape(b, self.dog_dim * self.history_length)

        # latest dog obs for fast orientation compensation
        cur_dog = x[:, -1, :self.dog_dim]
        base_ang_vel = cur_dog[:, 3:6]
        projected_gravity = cur_dog[:, 6:9]

        # arm history
        arm_hist = x[:, :, self.dog_dim:self.dog_dim + self.arm_dim]

        # latest arm obs
        arm = arm_hist[:, -1, :]

        arm_pos = arm[:, 0:6]
        arm_vel = arm[:, 6:12]
        arm_cmd = arm[:, 12:18]
        arm_pose_error = arm[:, 18:24]
        prev_action = arm[:, 24:30]

        arm_state_terms = [arm_pos, arm_vel, prev_action]
        if self.ik_dim > 0:
            arm_state_terms.append(arm[:, 30:30 + self.ik_dim])   # 第四章：IK 名义解偏差
        arm_state = torch.cat(arm_state_terms, dim=-1)

        # A: only one-step orientation dynamics, no GRU/TCN
        pos_err_now = arm_pose_error[:, 0:3]
        rot_err_now = arm_pose_error[:, 3:6]
        target_rpy = arm_cmd[:, 3:6]

        if self.history_length > 1:
            pos_err_prev = arm_hist[:, -2, 18:21]
            rot_err_prev = arm_hist[:, -2, 21:24]
            pos_err_delta_now = pos_err_now - pos_err_prev
            rot_err_delta_now = rot_err_now - rot_err_prev
        else:
            pos_err_delta_now = torch.zeros_like(pos_err_now)
            rot_err_delta_now = torch.zeros_like(rot_err_now)

        pos_err_delta_now = torch.clamp(pos_err_delta_now, -1.0, 1.0)
        rot_err_delta_now = torch.clamp(rot_err_delta_now, -1.0, 1.0)

        # 位置通路输入：误差 + 误差变化率，与 ori 通路对称
        pos_input = torch.cat([pos_err_now, pos_err_delta_now], dim=-1)  # 6

        ori_fast_input = torch.cat(
            [
                rot_err_now,          # 3
                rot_err_delta_now,    # 3
                target_rpy,           # 3
                arm_vel,              # 6
                prev_action,          # 6
                base_ang_vel,         # 3
                projected_gravity,    # 3
            ],
            dim=-1,
        )  # total 27

        # 全部历史帧的 pose error（pos 3 + ori 3），编码误差演化轨迹
        arm_err_hist = arm_hist[:, :, 18:24].reshape(b, -1)  # 6 * history_length

        return dog_input, arm_state, arm_cmd, pos_input, ori_fast_input, arm_err_hist

    def encode_dog(self, observations):
        b = observations.shape[0]
        x = observations.view(b, self.history_length, self.num_one_step_obs)

        dog_hist = x[:, :, :self.dog_dim]
        dog_input = dog_hist.reshape(b, self.dog_dim * self.history_length)

        dog_z = self.dog_prior(dog_input)
        return dog_z

    def encode_actor_features(self, observations):
        dog_input, arm_state, arm_cmd, pos_input, ori_fast_input, arm_err_hist = self.split_actor_obs(observations)

        dog_z = self.dog_prior(dog_input)

        arm_z = self.arm_state_encoder(arm_state)
        cmd_z = self.arm_cmd_encoder(arm_cmd)

        pos_z = self.pos_encoder(pos_input)

        ori_z = self.ori_fast_encoder(ori_fast_input)

        hist_z = self.arm_err_hist_encoder(arm_err_hist)

        actor_feat = self.pre_fusion(
            torch.cat([arm_z, pos_z, ori_z, cmd_z, hist_z], dim=-1)
        )

        actor_feat = self.base_film(actor_feat, dog_z)

        return actor_feat, dog_z, ori_z

    def update_distribution(self, observations):
        actor_feat, dog_z, ori_z = self.encode_actor_features(observations)

        main_action = self.main_action_head(actor_feat)

        ori_action = self.ori_action_head(
            torch.cat([ori_z, actor_feat, dog_z], dim=-1)
        )

        dog_bias = self.dog_action_bias(dog_z)

        mean = (
            main_action
            + self.ori_action_scale * ori_action
            + self.dog_prior_scale * dog_bias
        )

        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actor_feat, dog_z, ori_z = self.encode_actor_features(observations)

        main_action = self.main_action_head(actor_feat)

        ori_action = self.ori_action_head(
            torch.cat([ori_z, actor_feat, dog_z], dim=-1)
        )

        dog_bias = self.dog_action_bias(dog_z)

        return (
            main_action
            + self.ori_action_scale * ori_action
            + self.dog_prior_scale * dog_bias
        )

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def auxiliary_loss(
        self,
        observations,
        actions,
        critic_observations,
        next_critic_observations,
    ):
        # B: 先关闭 auxiliary loss，避免辅助预测任务抢占姿态控制梯度
        zero = torch.zeros((), device=observations.device)
        return zero, zero.detach(), zero.detach()