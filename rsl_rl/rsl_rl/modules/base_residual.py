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


class BaseResidual(nn.Module):
    """底座顺应残差策略。

    冻结底座照常输出步态动作，本网络在其上叠加一个小的残差动作 Δq（12 个狗关节），
    让底座为了帮机械臂够到目标而主动下蹲/前倾/挪重心。

    输入 obs（默认 57 维）：
        dog 当前 one-step obs   45  （命令/角速度/重力/关节位/关节速/dog_action）
        arm command             6   （l,p,y,roll,pitch,yaw）
        arm se3 error           6   （位置误差 3 + 姿态误差 3）

    设计要点（与 arm 残差网络一致的稳健性约定）：
      - 输出层 zero-init：训练初期 Δq≈0，系统等价于纯冻结基线，不破坏已有步态
      - 动作 tanh 限幅到 ±residual_clip，防止把底座推到危险姿态
      - 探索噪声小，且 std 上限 clamp（外部 PPO 再对 log_std 做参数投影）
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs,
        num_actions=12,
        activation="elu",
        hidden_dims=(256, 256),
        init_noise_std=0.1,
        residual_clip=0.3,
        std_max=0.5,
    ):
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.residual_clip = residual_clip
        self.std_max = std_max

        act = get_activation(activation)
        layers = []
        last = num_obs
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(act)
            last = h
        self.trunk = nn.Sequential(*layers)
        self.action_head = nn.Linear(last, num_actions)

        self.log_std = nn.Parameter(torch.log(torch.ones(num_actions) * init_noise_std))

        self.distribution = None
        self._init_weights()

    def _init_weights(self):
        # 输出层零初始化：初始残差为 0，不扰动冻结底座
        nn.init.constant_(self.action_head.weight, 0.0)
        nn.init.constant_(self.action_head.bias, 0.0)

    @property
    def std(self):
        return torch.exp(self.log_std).clamp(1e-4, self.std_max)

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

    def _raw_mean(self, obs):
        feat = self.trunk(obs)
        # 限幅后的残差均值（动作单位）
        return self.residual_clip * torch.tanh(self.action_head(feat))

    def update_distribution(self, obs):
        mean = self._raw_mean(obs)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        # 采样后再夹一次，保证应用到环境的动作始终在限幅内
        return self.distribution.sample().clamp(-self.residual_clip, self.residual_clip)

    def act_inference(self, obs):
        return self._raw_mean(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
