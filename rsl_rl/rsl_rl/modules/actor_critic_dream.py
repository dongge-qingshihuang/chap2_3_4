import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def get_activation(act_name: str):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"invalid activation function: {act_name}")


class ActorCritic_dream(nn.Module):
    """
    初始化签名必须匹配：
      ActorCritic_dream(num_obs, num_critic_obs, num_one_step_obs, num_actions, **policy_cfg)

    - observations: [B, num_obs]（通常是 history stack flatten）
    - VAE encoder: observations(或传入 obs_hist) -> vel_pred + z
    - Decoder: [vel_pred, z] -> next_one_step_obs_pred [B, num_one_step_obs]
    - Actor input: [observations, vel_pred, z] -> action mean
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,            # self.env.num_obs
        num_critic_obs: int,           # num_critic_obs
        num_one_step_obs: int,         # self.env.num_one_step_obs
        num_actions: int,              # self.env.num_actions
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        # ---- VAE params ----
        latent_dim: int = 16,
        vel_dim: int = 3,
        vae_enc_hidden_dims=(256, 128),
        vae_dec_hidden_dims=(256, 128),
        **kwargs
    ):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        act_fn = get_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_actions = num_actions

        self.latent_dim = latent_dim
        self.vel_dim = vel_dim

        # =========================
        # VAE Encoder (shared trunk)
        # =========================
        enc_layers = []
        in_dim = num_actor_obs  # 这里直接用 num_obs（history stack flatten）
        for h in vae_enc_hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.vae_encoder = nn.Sequential(*enc_layers)

        # two heads
        self.vel_head = nn.Linear(in_dim, vel_dim)          # 速度估计（监督）
        self.mu_head = nn.Linear(in_dim, latent_dim)        # latent 均值
        self.logvar_head = nn.Linear(in_dim, latent_dim)    # latent logvar

        # =========================
        # VAE Decoder: [vel, z] -> next one-step obs
        # =========================
        dec_layers = []
        in_dim = vel_dim + latent_dim
        for h in vae_dec_hidden_dims:
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        dec_layers += [nn.Linear(in_dim, num_one_step_obs)]
        self.vae_decoder = nn.Sequential(*dec_layers)

        # =========================
        # Actor: [obs, vel, z] -> action mean
        # =========================
        mlp_input_dim_a = num_actor_obs + vel_dim + latent_dim
        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), act_fn]
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers += [nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]), act_fn]
        actor_layers += [nn.Linear(actor_hidden_dims[-1], num_actions)]
        self.actor = nn.Sequential(*actor_layers)

        # =========================
        # Critic: critic_obs -> value
        # =========================
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), act_fn]
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers += [nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]), act_fn]
        critic_layers += [nn.Linear(critic_hidden_dims[-1], 1)]
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action distribution
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    # -------------------------
    # VAE core
    # -------------------------
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, obs_hist: torch.Tensor):
        """
        obs_hist: [B, num_actor_obs]（默认直接传 observations）
        return: vel_pred [B,vel_dim], z [B,latent_dim], mu/logvar
        """
        h = self.vae_encoder(obs_hist)
        vel_pred = self.vel_head(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        return vel_pred, z, mu, logvar

    def decode(self, vel_pred: torch.Tensor, z: torch.Tensor):
        """
        return next_one_step_obs_pred: [B, num_one_step_obs]
        """
        x = torch.cat([vel_pred, z], dim=-1)
        return self.vae_decoder(x)

    def vae_forward(self, obs_hist: torch.Tensor):
        """
        训练 VAE 用：返回 vel_pred, z, next_one_step_obs_pred, mu, logvar
        """
        vel_pred, z, mu, logvar = self.encode(obs_hist)
        next_obs_pred = self.decode(vel_pred, z)
        return vel_pred, z, next_obs_pred, mu, logvar


    # -------------------------
    # RL API
    # -------------------------
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, observations: torch.Tensor, obs_hist: torch.Tensor = None):
        """
        observations: [B, num_actor_obs]
        obs_hist: 可选；不传则默认 obs_hist=observations
        """
        if obs_hist is None:
            obs_hist = observations

        vel_pred, z, _, _ = self.encode(obs_hist)

        actor_in = torch.cat([observations, vel_pred.detach(), z.detach()], dim=-1)
        mean = self.actor(actor_in)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs):
        # 兼容：你也可以传 obs_hist=...
        obs_hist = kwargs.get("obs_hist", None)
        self.update_distribution(observations, obs_hist=obs_hist)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, **kwargs):
        obs_hist = kwargs.get("obs_hist", observations)
        vel_pred, z, _, _ = self.encode(obs_hist)
        actor_in = torch.cat([observations, vel_pred.detach(), z.detach()], dim=-1)
        action = self.actor(actor_in)
        return action,action,action

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(critic_observations)
