import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic_attention(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_proprio,
                        num_recon,
                        history_len, 
                        num_modes = 5,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        dha_hidden_dims = [256, 64, 32],   # 保留参数但不再使用
                        tsdyn_hidden_dims = [256, 128, 64],
                        tsdyn_latent_dims = 32,
                        terrain_dim: int = 187,            # critic_obs 开头高程图长度
                        num_attn_heads: int = 4,           # 交叉注意力头数
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super(ActorCritic_attention, self).__init__()
        
        self.terrain_dim = terrain_dim
        self.clip_actions = 3
        self.num_proprio = num_proprio
        self.num_recon = num_recon
        self.history_len = history_len
        self.num_modes = num_modes
        self.tsdyn_latent_dims = tsdyn_latent_dims
        self.num_attn_heads = num_attn_heads

        # Proprio 编码器 latent 维度
        prop_latent_dims = 32
        prop_hidden_out  = 64

        num_his_obs = num_actor_obs

        # === Actor / Critic 输入维度 ===
        # tsdyn_latent_dims + num_proprio + 3 (vel) + z_prop
        mlp_input_dim_a = tsdyn_latent_dims + num_actor_obs + 3 + prop_latent_dims
        mlp_input_dim_c = num_critic_obs
 

        # ---------- Actor ----------
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(nn.ELU())
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))  # 输出动作均值 μ
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(nn.ELU())
        self.actor = nn.Sequential(*actor_layers)

        # ---------- Critic ----------
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(nn.ELU())
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(nn.ELU())
        self.critic = nn.Sequential(*critic_layers)

        # ---------- Proprio 编码器 ----------
        self.prop_encoder = ProprioEncoder(
            num_actor_obs=num_actor_obs,
            history_len=history_len,
            prop_hidden_out=prop_hidden_out,
            prop_latent_dims=prop_latent_dims
        )

        # ---------- TsDyn：VAE 模式（重建 obs/contact） ----------
        self.TsDyn_modules = nn.ModuleList()
        TsDyn_input_dim = num_his_obs
        TsDyn_output_dim = self.num_recon   # 例如下一时刻 obs_without_vel
        for _ in range(num_modes):
            self.TsDyn_modules.append(
                VAE(TsDyn_input_dim, TsDyn_output_dim,
                    tsdyn_hidden_dims, tsdyn_latent_dims,
                    self.history_len, kl_w=0.9, prior_mu=0,
                    num_modes=num_modes)
            )

        # ---------- 地形编码 & 聚类（用来产生伪标签） ----------
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
        # 一些 loss 系数，方便在 update 里用
        self.terrain_recon_coef   = 1.0   # 高程重建损失权重
        self.cluster_pull_coef    = 0.25  # z 拉向簇中心
        self.cluster_balance_coef = 0.10  # 簇频率均衡
        self.dha_sup_coef         = 1.0   # 若用簇标签监督 DHA / 注意力等

        # ---------- 交叉注意力路由参数 ----------
        # context = [obs, vel_pred] 维度
        self.ctx_dim  = num_actor_obs 
        self.mode_dim = tsdyn_latent_dims
        d_model = tsdyn_latent_dims
        assert d_model % self.num_attn_heads == 0, \
            "tsdyn_latent_dims 必须能被 num_attn_heads 整除"

        # Q: context → d_model
        # K,V: mode_rep → d_model
        self.q_proj   = nn.Linear(self.ctx_dim,  d_model)
        self.k_proj   = nn.Linear(self.mode_dim, d_model)
        self.v_proj   = nn.Linear(self.mode_dim, d_model)
        self.out_proj = nn.Linear(d_model, self.mode_dim)  # 回到 D

        # 用于日志 / 正则
        self.last_attn_weights = None   # [B, K]

        # ---------- 动作分布 ----------
        self.distribution = None
        self.std = nn.Parameter(torch.ones(num_actions) * 1.0)
        Normal.set_default_validate_args = False
        

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
         for idx, module in enumerate(mod for mod in sequential
                                      if isinstance(mod, nn.Linear))]

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
    #  内部：TsDyn + 多头交叉注意力路由
    # -------------------------------------------------
    def _cross_attend_modes(self, observations: torch.Tensor):
        """
        输入:
            observations: [B, num_actor_obs]
            vel_pre:      [B, 3]
        返回:
            agg_rep: [B, D]   聚合后的 TsDyn 表征
            attn_w:  [B, K]   每个样本对 K 个模式的权重
        """
        B = observations.size(0)

        # 1) K 个 TsDyn 模式表征 z_k: [B, K, D]
        reps = []
        for sub_net in self.TsDyn_modules:
            z_k = sub_net.get_representation(observations).unsqueeze(1)  # [B,1,D]
            reps.append(z_k)
        mode_reps = torch.cat(reps, dim=1)  # [B,K,D]
        B, K, D = mode_reps.shape

        # 2) 上下文 Q: [B, ctx_dim] → [B, d_model]
        context = observations  # [B, ctx_dim]
        Q = self.q_proj(context)                              # [B, d_model]

        # 3) K,V: [B,K,D] → [B,H,K,d_k]
        K_all = self.k_proj(mode_reps)  # [B,K,d_model]
        V_all = self.v_proj(mode_reps)  # [B,K,d_model]

        H = self.num_attn_heads
        d_model = Q.size(-1)
        d_k = d_model // H

        # Q: [B,H,d_k]
        Q = Q.view(B, H, d_k)
        # K,V: [B,H,K,d_k]
        K_all = K_all.view(B, K, H, d_k).transpose(1, 2)
        V_all = V_all.view(B, K, H, d_k).transpose(1, 2)

        # 4) scaled dot-product attention
        # attn_scores: [B,H,K]
        attn_scores = torch.einsum("bhd,bhkd->bhk", Q, K_all) / (d_k ** 0.5)
        attn_w = F.softmax(attn_scores, dim=-1)  # [B,H,K]

        # 5) 聚合 V: [B,H,K,d_k] → [B,H,d_k] → [B,d_model]
        agg = torch.einsum("bhk,bhkd->bhd", attn_w, V_all)  # [B,H,d_k]
        agg = agg.reshape(B, d_model)                      # [B,d_model]
        agg = self.out_proj(agg)                           # [B,D]

        # 多头权重取平均，给外部用 [B,K]
        attn_w_mean = attn_w.mean(dim=1)
        self.last_attn_weights = attn_w_mean

        return agg, attn_w_mean

    # -------------------------------------------------
    #  PPO 采样时的分布更新
    # -------------------------------------------------
    def update_distribution(self, observations):
        assert not torch.isnan(observations).any(), "Observations contain NaN values!"

        # Proprio & velocity
        z_prop, vel_pre = self.prop_encoder(observations)

        # TsDyn + 交叉注意力路由
        representation, attn_w = self._cross_attend_modes(
            observations
        )

        # Actor 输入：proprio 尾巴 + TsDyn 表征 + 速度 + proprio latent
        prop_tail = observations[:, -self.num_proprio:]
        actor_inp = torch.cat(
            (prop_tail, representation.detach(), vel_pre.detach(), z_prop),
            dim=-1
        )
        mean = self.actor(actor_inp)
        self.distribution = Normal(mean, 0. * mean + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample()
        return actions
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # -------------------------------------------------
    #  推理（确定性均值）
    # -------------------------------------------------
    def act_inference(self, observations):
        # 1) Proprio 编码（不希望梯度回传到 encoder）
        z_prop, vel_pre = self.prop_encoder(observations)
        z_prop  = z_prop.detach()
        vel_pre = vel_pre.detach()

        # 2) TsDyn + 交叉注意力（推理时也用，但整条路径 detach）
        representation, attn_w = self._cross_attend_modes(
            observations
        )
        representation = representation.detach()

        # 3) 组装 Actor 输入
        prop_tail = observations[:, -self.num_proprio:].detach()
        actor_inp = torch.cat((observations, representation, vel_pre, z_prop), dim=-1)

        mean = self.actor(actor_inp)
        return mean,attn_w,z_prop

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    # 保留旧接口名，方便外部直接调用
    def act_infer(self, observations):
        return self.act_inference(observations)
    

    def get_ts_representation(self, observations):
        """
        返回聚合后的时序潜表示 z（自监督 / 诊断使用）
        """
        z_prop, vel_pre = self.prop_encoder(observations)
        representation, attn_w = self._cross_attend_modes(
            observations
        )
        return representation


# ====================== 工具模块 ======================

class ProprioEncoder(nn.Module):
    def __init__(self, num_actor_obs: int, history_len: int,
                 prop_hidden_out: int = 64,  # CNN1dEstimator 的 output_size
                 prop_latent_dims: int = 32):
        super().__init__()
        assert num_actor_obs % history_len == 0
        self.history_len = history_len
        self.per_step_dim = num_actor_obs // history_len

        self.backbone = CNN1dEstimator(
            nn.ReLU(), input_size=self.per_step_dim,
            tsteps=history_len, output_size=prop_hidden_out
        )
        self.proj     = nn.Linear(prop_hidden_out, prop_latent_dims)  # z_prop
        self.vel_head = nn.Linear(prop_hidden_out, 3)                 # 速度预测 v̂

    def forward(self, flat_obs):  # 返回 (z_prop, vel_pred)
        B = flat_obs.size(0)
        x = flat_obs.view(B, self.history_len, self.per_step_dim)
        h = self.backbone(x)                 # [B, prop_hidden_out]
        z_prop   = self.proj(h)              # [B, d_z]
        vel_pred = self.vel_head(h)          # [B, 3]
        return z_prop, vel_pred


class CNN1dEstimator(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super(CNN1dEstimator, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn,
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size,
                          kernel_size = 8, stride = 4), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size,
                          kernel_size = 5, stride = 1), self.activation_fn,
                nn.Conv1d(in_channels = channel_size, out_channels = channel_size,
                          kernel_size = 5, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size,
                          kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size,
                          kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size,
                          kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size,
                          kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn
        )
        

    def forward(self, obs):
        # obs: [B, T, per_step_dim]
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))  # [B*T, 3*ch]
        output = self.conv_layers(
            projection.reshape([nd, T, -1]).permute((0, 2, 1))
        )
        output = self.linear_output(output)
        return output
    

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dims, 
                 hidden_dims, 
                 latent_dim = 16,
                 history_len = 20,
                 feet_contact_dim = 4,
                 kl_w = 0.1,
                 prior_mu=None,
                 num_modes: int = 8   # 地形类别数 = num_modes
                 ):
        """
        Args:
            input_dim (int): 输入维度（history 展平）
            output_dims (int): 重建的观测维度
            hidden_dims (list of int): MLP hidden 层
            latent_dim (int): 潜空间维度
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim 
        self.output_dims = output_dims
        self.per_step_dim = int(input_dim // history_len)
        self.history_len = history_len

        self.encoder = CNN1dEstimator(
            nn.ReLU(),
            int(input_dim // history_len),
            history_len,
            hidden_dims[-1]
        )

        # Latent space
        self.fc_mu     = nn.Linear(hidden_dims[-1] , latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] , latent_dim)
        
        # Decoder (重建观测)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], output_dims))
        self.decoder = nn.Sequential(*decoder_layers)

        # Feet contact decoder
        FC_decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            FC_decoder_layers.append(nn.Linear(prev_dim, h_dim))
            FC_decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        FC_decoder_layers.append(nn.Linear(hidden_dims[0], feet_contact_dim))
        FC_decoder_layers.append(nn.Sigmoid())
        self.FC_decoder = nn.Sequential(*FC_decoder_layers)

        self.prior_mu = prior_mu
        self.kl_w = kl_w

        # ===== 新增：地形分类头 z -> num_modes logits =====
        self.ter_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], num_modes)   # num_modes = 路由模式 / 地形簇数
        )

    # --- encode ---
    def encode(self, x):
        B = x.shape[0]; T = self.history_len
        x_steps = x.view(B, T, self.per_step_dim)     # [B,T,p]
        h = self.encoder(x_steps)                     # [B, hidden_dims[-1]]
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
        ter_logits = self.ter_head(z)          # 地形 logits: [B, num_modes]
        # 返回 5 项，方便 update 解包
        return recon_x, recon_contact, ter_logits, mu, logvar



class TerrainEncoder(nn.Module):
    """
    特权地形编码器：输入展平高程图向量，输出 latent；可选返回重建。
    用于 update 里做 EMA-KMeans 聚类，产生 terrain 伪标签。
    """
    def __init__(self, input_dim: int, ter_hidden_out: int = 128, ter_latent_dims: int = 32):
        super().__init__()
        # encoder: heights -> z
        self.enc = nn.Sequential(
            nn.Linear(input_dim, ter_hidden_out), nn.ReLU(),
            nn.Linear(ter_hidden_out, ter_latent_dims)
        )
        # decoder: z -> heights_recon
        self.dec = nn.Sequential(
            nn.Linear(ter_latent_dims, ter_hidden_out), nn.ReLU(),
            nn.Linear(ter_hidden_out, input_dim)   # 若 heights ∈ [0,1]，可在末尾加 Sigmoid()
        )

    def forward(self, heights_vec: torch.Tensor, return_recon: bool = False):
        z = self.enc(heights_vec)
        if return_recon:
            recon = self.dec(z)
            return z, recon
        return z


class EmaKMeans(nn.Module):
    """
    EMA-KMeans：对 terrain latent 做在线聚类，产生离散簇标签作为伪 terrain label。
    """
    def __init__(self, K: int, D: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K, self.D, self.decay, self.eps = K, D, decay, eps
        self.register_buffer("centroids", torch.randn(K, D))
        self.register_buffer("ema_count", torch.zeros(K))
        self.register_buffer("ema_sum", torch.zeros(K, D))

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, one_hot: torch.Tensor):
        # z: [B, D], one_hot: [B, K]
        one_hot = one_hot.to(self.ema_count.dtype)

        count = one_hot.sum(0)         # [K]
        summ  = one_hot.t() @ z        # [K, D]

        self.ema_count.mul_(self.decay).add_(count, alpha=(1.0 - self.decay))
        self.ema_sum.mul_(self.decay).add_(summ,  alpha=(1.0 - self.decay))

        denom = self.ema_count.clamp_min(self.eps).unsqueeze(1)  # [K, 1]
        self.centroids.copy_(self.ema_sum / denom)

    def forward(self, z: torch.Tensor):
        # 距离矩阵 d: [B,K]
        d = (z.pow(2).sum(1, keepdim=True)
             - 2 * z @ self.centroids.t()
             + self.centroids.pow(2).sum(1, keepdim=True).t())
        indices = d.argmin(dim=1)                               # [B]
        one_hot = F.one_hot(indices, num_classes=self.K).type_as(z)  # [B,K]

        with torch.no_grad():
            self._ema_update(z.detach(), one_hot.detach())

        # 拉近当前质心（对 z 有梯度）
        chosen_c = self.centroids[indices]                      # [B,D]
        pull_loss = F.mse_loss(z, chosen_c)

        return indices, one_hot, pull_loss




class InferenceActor(nn.Module):
    def __init__(self, actor_module):
        super().__init__()
        self.actor_module = actor_module
        
    def forward(self, x):
        with torch.no_grad():  # 推理
            actions= self.actor_module.act_inference(x)  # 关键：不要 self.actor_module(x_norm)
            return actions

class InferenceActor_normalizer(nn.Module):
    def __init__(self, actor_module, obs_normalizer):
        super().__init__()
        self.actor_module = actor_module
        self.obs_normalizer = obs_normalizer
        
    def forward(self, x):
        with torch.no_grad():  # 推理
            n,d = x.size()
            x_norm = self.obs_normalizer(x.reshape(-1,self.actor_module.num_proprio)).reshape(n,d)
            actions= self.actor_module.act_inference(x_norm)  # 关键：不要 self.actor_module(x_norm)
            return actions