import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# =========================================================
#  纯 MLP 版 History Encoder：替代 CNN1dEstimator + Conv1d
# =========================================================
class MLPHistoryEncoder(nn.Module):
    """
    输入可以是：
      - [B, T, per_step_dim]  (会自动 flatten)
      - [B, T*per_step_dim]   (直接用)
    输出: [B, output_size]
    """
    def __init__(self, input_dim: int, output_size: int,
                 hidden_dims=(256, 256), activation=nn.ReLU()):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), activation]
            last = h
        layers += [nn.Linear(last, output_size), activation]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:  # [B,T,p] -> [B,T*p]
            x = x.reshape(x.size(0), -1)
        return self.net(x)


# =========================================================
#  Proprio Encoder（MLP版）
# =========================================================
class ProprioEncoder(nn.Module):
    def __init__(self,
                 num_actor_obs: int,
                 history_len: int,
                 prop_hidden_out: int = 64,
                 prop_latent_dims: int = 32,
                 payload_dim: int = 0):
        super().__init__()
        # 这里 history_len 只是为了语义一致（你后面可能仍会用到）
        assert num_actor_obs % history_len == 0
        self.history_len = history_len
        self.per_step_dim = num_actor_obs // history_len

        # 纯 MLP：直接对 flat history obs 编码
        self.backbone = MLPHistoryEncoder(
            input_dim=num_actor_obs,
            output_size=prop_hidden_out,
            hidden_dims=(256, 256),
            activation=nn.ReLU()
        )
        self.proj = nn.Linear(prop_hidden_out, prop_latent_dims)   # z_prop
        self.vel_head = nn.Linear(prop_hidden_out, 3)              # v_hat

        # 第二章黑盒负载感知：与速度头同构，从本体历史估计背上负载的等效状态。
        # payload_dim=0 时不创建，行为与原来完全一致。
        self.payload_dim = payload_dim
        self.payload_head = nn.Linear(prop_hidden_out, payload_dim) if payload_dim > 0 else None

    def forward_full(self, flat_obs):
        """返回 (z_prop, vel_pred, payload_pred)，payload_pred 在未启用时为 None。"""
        h = self.backbone(flat_obs)      # [B, prop_hidden_out]
        z_prop = self.proj(h)            # [B, prop_latent_dims]
        vel_pred = self.vel_head(h)      # [B, 3]
        payload_pred = self.payload_head(h) if self.payload_head is not None else None
        return z_prop, vel_pred, payload_pred

    def forward(self, flat_obs):
        z_prop, vel_pred, _ = self.forward_full(flat_obs)
        return z_prop, vel_pred


# =========================================================
#  VAE（MLP版）
# =========================================================
class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dims,
                 hidden_dims,
                 latent_dim=16,
                 history_len=20,
                 feet_contact_dim=4,
                 kl_w=0.1,
                 prior_mu=None,
                 num_modes: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.history_len = history_len
        self.kl_w = kl_w
        self.prior_mu = prior_mu

        # MLP encoder：x(flat) -> hidden_dims[-1]
        enc_out = hidden_dims[-1]
        enc_hiddens = tuple(hidden_dims[:-1]) if len(hidden_dims) > 1 else (256, 256)

        self.encoder = MLPHistoryEncoder(
            input_dim=input_dim,
            output_size=enc_out,
            hidden_dims=enc_hiddens,
            activation=nn.ReLU()
        )

        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)

        # Decoder：z -> recon obs
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], output_dims))
        self.decoder = nn.Sequential(*decoder_layers)

        # Feet contact decoder：z -> contact prob
        fc_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            fc_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        fc_layers += [nn.Linear(hidden_dims[0], feet_contact_dim), nn.Sigmoid()]
        self.FC_decoder = nn.Sequential(*fc_layers)

        # Terrain logits head
        self.ter_head = nn.Sequential(
            nn.Linear(latent_dim, enc_out),
            nn.ReLU(),
            nn.Linear(enc_out, num_modes)
        )

    def encode(self, x):
        # x: [B, input_dim] (flat)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_representation(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        contact_hat = self.FC_decoder(z)
        return x_hat, contact_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, recon_contact = self.decode(z)
        ter_logits = self.ter_head(z)
        return recon_x, recon_contact, ter_logits, mu, logvar


# =========================================================
#  Terrain Encoder / EMA-KMeans（原样保留，已是 MLP）
# =========================================================
class TerrainEncoder(nn.Module):
    def __init__(self, input_dim: int, ter_hidden_out: int = 128, ter_latent_dims: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, ter_hidden_out), nn.ReLU(),
            nn.Linear(ter_hidden_out, ter_latent_dims)
        )
        self.dec = nn.Sequential(
            nn.Linear(ter_latent_dims, ter_hidden_out), nn.ReLU(),
            nn.Linear(ter_hidden_out, input_dim)
        )

    def forward(self, heights_vec: torch.Tensor, return_recon: bool = False):
        z = self.enc(heights_vec)
        if return_recon:
            recon = self.dec(z)
            return z, recon
        return z


class EmaKMeans(nn.Module):
    def __init__(self, K: int, D: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K, self.D, self.decay, self.eps = K, D, decay, eps
        self.register_buffer("centroids", torch.randn(K, D))
        self.register_buffer("ema_count", torch.zeros(K))
        self.register_buffer("ema_sum", torch.zeros(K, D))

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, one_hot: torch.Tensor):
        one_hot = one_hot.to(self.ema_count.dtype)
        count = one_hot.sum(0)         # [K]
        summ = one_hot.t() @ z         # [K, D]
        self.ema_count.mul_(self.decay).add_(count, alpha=(1.0 - self.decay))
        self.ema_sum.mul_(self.decay).add_(summ, alpha=(1.0 - self.decay))
        denom = self.ema_count.clamp_min(self.eps).unsqueeze(1)
        self.centroids.copy_(self.ema_sum / denom)

    def forward(self, z: torch.Tensor):
        d = (z.pow(2).sum(1, keepdim=True)
             - 2 * z @ self.centroids.t()
             + self.centroids.pow(2).sum(1, keepdim=True).t())
        indices = d.argmin(dim=1)
        one_hot = F.one_hot(indices, num_classes=self.K).type_as(z)

        with torch.no_grad():
            self._ema_update(z.detach(), one_hot.detach())

        chosen_c = self.centroids[indices]
        pull_loss = F.mse_loss(z, chosen_c)
        return indices, one_hot, pull_loss


# =========================================================
#  ActorCritic_attention（MLP版：所有卷积已移除）
# =========================================================
class Actor_mulCritic_attention1(nn.Module):
    is_recurrent = False

    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_proprio,
                 num_recon,
                 history_len,
                 num_modes=8,
                actor_hidden_dims = [512, 256, 128],
                critic_hidden_dims = [512, 256, 128],
                 dha_hidden_dims=[256, 64, 32],       # 保留参数但不再使用
                 tsdyn_hidden_dims=[256, 128, 64],
                 tsdyn_latent_dims=32,
                 terrain_dim: int = 187,
                 num_attn_heads: int = 4,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.terrain_dim = terrain_dim
        self.clip_actions = 3
        self.num_proprio = num_proprio
        self.num_recon = num_recon
        self.history_len = history_len
        self.num_modes = num_modes
        self.tsdyn_latent_dims = tsdyn_latent_dims
        self.num_attn_heads = num_attn_heads

        # Proprio latent 维度
        prop_latent_dims = 32
        prop_hidden_out = 64

        # 第二章黑盒负载感知：估计出的负载状态也送进 actor（本体观测本身不变）
        self.payload_dim = int(kwargs.get("payload_dim", 0))

        # Actor 输入维度：observations(flat) + representation + vel(3) + payload + z_prop
        mlp_input_dim_a = tsdyn_latent_dims + num_actor_obs + 3 + self.payload_dim + prop_latent_dims
        mlp_input_dim_c = num_critic_obs

        # ---------- Actor ----------
        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), nn.ELU()]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(nn.ELU())
        self.actor = nn.Sequential(*actor_layers)

        # ---------- Critic: shared trunk + 3 heads ----------
        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), nn.ELU()]
        for l in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
            critic_layers.append(nn.ELU())
        self.critic_trunk = nn.Sequential(*critic_layers)

        last_dim = critic_hidden_dims[-1]
        self.critic_task = nn.Linear(last_dim, 1)
        self.critic_stab = nn.Linear(last_dim, 1)
        self.critic_energy = nn.Linear(last_dim, 1)



        # ---------- Proprio 编码器（MLP版） ----------
        self.prop_encoder = ProprioEncoder(
            num_actor_obs=num_actor_obs,
            history_len=history_len,
            prop_hidden_out=prop_hidden_out,
            prop_latent_dims=prop_latent_dims,
            payload_dim=self.payload_dim
        )

        # ---------- TsDyn VAE 模式（MLP版） ----------
        self.TsDyn_modules = nn.ModuleList()
        TsDyn_input_dim = num_actor_obs
        TsDyn_output_dim = self.num_recon
        for _ in range(num_modes):
            self.TsDyn_modules.append(
                VAE(TsDyn_input_dim, TsDyn_output_dim,
                    tsdyn_hidden_dims, tsdyn_latent_dims,
                    self.history_len, kl_w=0.9, prior_mu=0,
                    num_modes=num_modes)
            )

        # ---------- terrain encoder & cluster ----------
        self.terrain_encoder = TerrainEncoder(
            input_dim=terrain_dim,
            ter_hidden_out=128,
            ter_latent_dims=tsdyn_latent_dims
        )
        self.terrain_clusterer = EmaKMeans(
            K=num_modes,
            D=tsdyn_latent_dims,
            decay=0.99
        )
        self.terrain_recon_coef = 1.0
        self.cluster_pull_coef = 0.25
        self.cluster_balance_coef = 0.10
        self.dha_sup_coef = 1.0

        # ---------- cross-attn routing ----------
        self.ctx_dim = num_actor_obs
        self.mode_dim = tsdyn_latent_dims
        d_model = tsdyn_latent_dims
        assert d_model % self.num_attn_heads == 0, "tsdyn_latent_dims 必须能被 num_attn_heads 整除"

        self.q_proj = nn.Linear(self.ctx_dim, d_model)
        self.k_proj = nn.Linear(self.mode_dim, d_model)
        self.v_proj = nn.Linear(self.mode_dim, d_model)
        self.out_proj = nn.Linear(d_model, self.mode_dim)

        self.last_attn_weights = None  # [B,K]

        # ---------- action distribution ----------
        self.distribution = None
        self.std = nn.Parameter(torch.ones(num_actions) * 1.0)
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
         for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # -------------------------------------------------
    # TsDyn + multihead cross-attn
    # -------------------------------------------------
    def _cross_attend_modes(self, observations: torch.Tensor):
        B = observations.size(0)

        # 1) mode reps: [B,K,D]
        reps = []
        for sub_net in self.TsDyn_modules:
            z_k = sub_net.get_representation(observations).unsqueeze(1)  # [B,1,D]
            reps.append(z_k)
        mode_reps = torch.cat(reps, dim=1)  # [B,K,D]
        _, K, D = mode_reps.shape

        # 2) Q: [B,ctx] -> [B,d_model]
        Q = self.q_proj(observations)

        # 3) K,V: [B,K,D] -> [B,K,d_model]
        K_all = self.k_proj(mode_reps)
        V_all = self.v_proj(mode_reps)

        H = self.num_attn_heads
        d_model = Q.size(-1)
        d_k = d_model // H

        # reshape
        Q = Q.view(B, H, d_k)                          # [B,H,d_k]
        K_all = K_all.view(B, K, H, d_k).transpose(1, 2)  # [B,H,K,d_k]
        V_all = V_all.view(B, K, H, d_k).transpose(1, 2)  # [B,H,K,d_k]

        # attention
        attn_scores = torch.einsum("bhd,bhkd->bhk", Q, K_all) / (d_k ** 0.5)
        attn_w = F.softmax(attn_scores, dim=-1)           # [B,H,K]

        agg = torch.einsum("bhk,bhkd->bhd", attn_w, V_all)  # [B,H,d_k]
        agg = agg.reshape(B, d_model)                       # [B,d_model]
        agg = self.out_proj(agg)                            # [B,D]

        attn_w_mean = attn_w.mean(dim=1)  # [B,K]
        self.last_attn_weights = attn_w_mean
        return agg, attn_w_mean

    # -------------------------------------------------
    # PPO sampling distribution
    # -------------------------------------------------
    def update_distribution(self, observations):
        assert not torch.isnan(observations).any(), "Observations contain NaN values!"

        z_prop, vel_pre, payload_pre = self.prop_encoder.forward_full(observations)
        representation, _ = self._cross_attend_modes(observations)

        parts = [observations, representation.detach(), vel_pre.detach()]
        if payload_pre is not None:
            parts.append(payload_pre.detach())
        parts.append(z_prop.detach())

        mean = self.actor(torch.cat(parts, dim=-1))
        self.distribution = Normal(mean, 0. * mean + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # -------------------------------------------------
    # inference (deterministic mean)
    # -------------------------------------------------
    def act_inference(self, observations):
        z_prop, vel_pre, payload_pre = self.prop_encoder.forward_full(observations)
        z_prop = z_prop.detach()
        vel_pre = vel_pre.detach()

        representation, attn_w_mean = self._cross_attend_modes(observations)
        representation = representation.detach()

        # 这里统一用 observations（全量 flat history）保证维度一致
        parts = [observations, representation, vel_pre]
        if payload_pre is not None:
            parts.append(payload_pre.detach())
        parts.append(z_prop)

        mean = self.actor(torch.cat(parts, dim=-1))
        return mean, attn_w_mean, mean

    # def evaluate(self, critic_observations, **kwargs):
    #     return self.critic(critic_observations)


    def evaluate(self, critic_observations, **kwargs):
        feat = self.critic_trunk(critic_observations)  # (B, last_dim)

        v_task = self.critic_task(feat)       # (B,1)
        v_stab = self.critic_stab(feat)       # (B,1)
        v_energy = self.critic_energy(feat)   # (B,1)

        # 返回 (B,3): [task, stab, energy]
        return torch.cat([v_task, v_stab, v_energy], dim=-1)



    # 兼容旧接口名
    def act_infer(self, observations):
        return self.act_inference(observations)

    def get_ts_representation(self, observations):
        z_prop, vel_pre = self.prop_encoder(observations)
        representation, attn_w = self._cross_attend_modes(observations)
        return representation
