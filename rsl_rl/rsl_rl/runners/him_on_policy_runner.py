import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO, HIMPPO
from rsl_rl.modules import HIMActorCritic, ActorCritic,Actor_mulCritic_attention1,ActorCriticArmServo,ActorCriticArmServoDogPrior,ActorCriticArmServoDogPrior_nofilm
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic_dream
from rsl_rl.modules import ActorCritic_attention
from rsl_rl.modules import ActorCritic_attention1
from rsl_rl.modules import (
    EmpiricalDiscountedVariationNormalization,
    EmpiricalHistoryNormalization,
    EmpiricalNormalization
)
from rsl_rl.modules import BaseResidual
from rsl_rl.modules import TerrainMoeActorCritic


class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        self._pretrained_obs_normalizer_state = None

        self.policy_class_name = self.cfg["policy_class_name"]
        actor_critic_class = eval(self.policy_class_name)

        # 这一轮先保持与当前 env 接口一致
        # 后面 env 改成 arm policy 真控机械臂后，再切到 self.env.num_arm_actions
        self.train_action_dim = self.env.num_actions

        if self.policy_class_name == "ActorCritic":
            actor_kwargs = {
                key: self.policy_cfg[key]
                for key in (
                    "actor_hidden_dims",
                    "critic_hidden_dims",
                    "activation",
                    "init_noise_std",
                )
                if key in self.policy_cfg
            }
            actor_critic = actor_critic_class(
                self.env.num_obs,
                num_critic_obs,
                self.train_action_dim,
                **actor_kwargs
            ).to(self.device)
        elif self.policy_class_name == "TerrainMoeActorCritic":
            actor_critic = actor_critic_class(
                self.env.num_obs,
                num_critic_obs,
                self.train_action_dim,
                self.env.num_one_step_obs,
                self.env.num_rec,
                self.env.history_length,
                **self.policy_cfg
            ).to(self.device)
            pretrained_checkpoint = self.cfg.get(
                "pretrained_base_checkpoint"
            )
            if pretrained_checkpoint and not self.cfg.get("resume", False):
                self._initialize_trainable_policy_from_pretrained(
                    actor_critic, pretrained_checkpoint
                )
            actor_critic.configure_finetune(
                mode=self.cfg.get(
                    "pretrained_policy_finetune_mode", "partial"
                ),
                expert_freeze=self.cfg.get("expert_freeze", True),
                gate_lr_multiplier=self.cfg.get(
                    "gate_lr_multiplier", 0.25
                ),
                adapter_lr_multiplier=self.cfg.get(
                    "adapter_lr_multiplier", 0.25
                ),
                actor_lr_multiplier=self.cfg.get(
                    "actor_lr_multiplier", 1.0
                ),
                expert_lr_multiplier=self.cfg.get(
                    "expert_lr_multiplier", 0.1
                ),
                freeze_actor=self.cfg.get(
                    "freeze_pretrained_actor", False
                ),
            )
        elif self.policy_class_name.startswith("ActorCriticArmServoDogPrior"):
            # 这两个网络的签名是 (..., num_one_step_obs, history_length, activation)，
            # 与下面的六位置参数形式不兼容，必须按关键字传
            actor_critic = actor_critic_class(
                self.env.num_obs,
                num_critic_obs,
                self.train_action_dim,
                num_one_step_obs=self.env.num_one_step_obs,
                history_length=self.env.history_length,
                **self.policy_cfg
            ).to(self.device)
        else:
            actor_critic = actor_critic_class(
                self.env.num_obs,
                num_critic_obs,
                self.train_action_dim,
                self.env.num_one_step_obs,
                self.env.num_one_step_obs,
                self.env.history_length,
                payload_dim=int(getattr(self.env.cfg.arm.payload, "priv_dim", 0)),
                **self.policy_cfg
            ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # 多评论家要求 critic 的输出头数与 value_dim 一致，否则后面只会抛出看不懂的广播错误
        if self.alg.value_dim > 1:
            with torch.no_grad():
                probe = actor_critic.evaluate(torch.zeros(2, num_critic_obs, device=self.device))
            if probe.shape[-1] != self.alg.value_dim:
                raise ValueError(
                    f"value_dim={self.alg.value_dim} 但 {self.policy_class_name}.evaluate() "
                    f"只输出 {probe.shape[-1]} 维。多评论家需要三头 critic，"
                    f"请用 Actor_mulCritic_attention1，或把 MULTI_CRITIC 关掉。"
                )

        # 第二章负载感知：把特权观测里负载真值的位置交给算法，用于估计器的监督损失
        payload_slice = getattr(self.env, "payload_priv_slice", None)
        if payload_slice is not None and getattr(actor_critic, "payload_dim", 0) > 0:
            self.alg.payload_slice = payload_slice
            self.alg.payload_loss_coef = float(
                getattr(self.env.cfg.arm.payload, "estimator_loss_coef", 1.0)
            )
            print(f"[Payload] 黑盒估计已启用：特权真值切片 {payload_slice}, "
                  f"loss_coef={self.alg.payload_loss_coef}")
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalHistoryNormalization(
                shape=[self.env.num_one_step_obs], until=1.0e8
            ).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=[num_critic_obs], until=1.0e8
            ).to(self.device)
            if self._pretrained_obs_normalizer_state is not None:
                with torch.inference_mode():
                    self.obs_normalizer.load_state_dict(
                        self._pretrained_obs_normalizer_state
                    )
                print(
                    "[PretrainedPolicy] initialized observation normalizer "
                    "from checkpoint"
                )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)


        self.use_frozen_base_policy = self.cfg.get("use_frozen_base_policy", False)
        self.base_policy = None
        self.base_obs_normalizer = torch.nn.Identity().to(self.device)

        if self.use_frozen_base_policy:
            self._build_and_load_base_policy(train_cfg)

        # --- 底座顺应残差（联合 fine-tune）：冻结底座照旧，额外训一个残差网络 ---
        self.train_base = False
        base_obs_shape = None
        base_actions_shape = None
        jf = getattr(getattr(self.env.cfg, "arm", None), "joint_finetune", None)
        if jf is not None and getattr(jf, "enabled", False) and self.use_frozen_base_policy:
            self._build_base_residual(jf)
            self.train_base = True
            base_obs_shape = [self.env.num_base_residual_obs]
            base_actions_shape = [self.env.num_dog_actions]

        self.residual_context_dim = int(
            getattr(
                self.alg.actor_critic,
                "base_action_residual_context_dim",
                0,
            )
            if getattr(
                self.alg.actor_critic,
                "base_action_residual_enabled",
                False,
            )
            else 0
        )
        self.residual_context_shape = (
            [self.residual_context_dim]
            if self.residual_context_dim > 0
            else None
        )
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.train_action_dim],
            base_obs_shape=base_obs_shape,
            base_actions_shape=base_actions_shape,
            residual_context_shape=self.residual_context_shape,
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def _initialize_trainable_policy_from_pretrained(
        self, actor_critic, checkpoint_path
    ):
        """Initialize Chapter 4 PPO actor from the Chapter 3 base checkpoint.

        The pretrained checkpoint is first loaded into its original
        single-critic architecture with ``strict=True``. Matching actor and
        perception parameters are then copied into the new trainable policy;
        only explicitly new Chapter 4 modules are initialized fresh.
        """
        checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        expected_version = self.cfg.get(
            "pretrained_policy_architecture_version",
            "go2_terrain_moe_v3_mlp_temporal_tcn_gate",
        )
        if checkpoint.get("architecture_version") != expected_version:
            raise RuntimeError(
                "Pretrained base architecture mismatch: "
                f"expected {expected_version}, "
                f"got {checkpoint.get('architecture_version')}"
            )
        self._pretrained_obs_normalizer_state = (
            checkpoint.get("obs_normalizer_state_dict")
            or checkpoint.get("obs_norm_state_dict")
        )
        if self._pretrained_obs_normalizer_state is not None:
            print(
                "[PretrainedPolicy] found pretrained observation "
                "normalization statistics"
            )

        base_policy_cfg = self.cfg.get("base_policy", {})
        pretrained_model = TerrainMoeActorCritic(
            45 * 20,
            242,
            12,
            45,
            None,
            20,
            **base_policy_cfg
        ).to(self.device)
        pretrained_model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        source_state = pretrained_model.state_dict()
        destination_state = actor_critic.state_dict()
        allowed_new_prefixes = (
            "critic.",
            "critic_trunk.",
            "critic_heads.",
            "payload_estimator.",
            "load_weight_net.",
            "base_action_residual.",
        )
        unexpected_new = [
            key
            for key in destination_state
            if key not in source_state
            and not key.startswith(allowed_new_prefixes)
        ]
        if unexpected_new:
            raise RuntimeError(
                "Unexpected new parameters in Chapter 4 policy: "
                f"{unexpected_new[:10]}"
            )
        transferred = {}
        for key, value in source_state.items():
            if key.startswith("critic."):
                continue
            if key not in destination_state:
                raise RuntimeError(
                    f"Pretrained parameter missing from trainable policy: {key}"
                )
            if destination_state[key].shape != value.shape:
                raise RuntimeError(
                    "Pretrained parameter shape mismatch for "
                    f"{key}: checkpoint={tuple(value.shape)}, "
                    f"policy={tuple(destination_state[key].shape)}"
                )
            transferred[key] = value
        actor_critic.load_state_dict(transferred, strict=False)
        loaded_state = actor_critic.state_dict()
        for key, value in transferred.items():
            if not torch.equal(loaded_state[key], value):
                raise RuntimeError(
                    f"Pretrained parameter verification failed: {key}"
                )
        actor_critic._pretrained_init_checkpoint = checkpoint_path
        actor_critic._pretrained_initialized_keys = len(transferred)
        actor_critic._pretrained_architecture_version = expected_version
        print(
            "[PretrainedPolicy] strict-validated original checkpoint and "
            f"initialized {len(transferred)} trainable-policy parameters"
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.alg.actor_critic.train()

        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            self.env.ppo_it = it
            self.alg.update_entropy_coef(it, tot_iter)

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self.use_frozen_base_policy:
                        dog_actions = self._infer_dog_actions()
                        self.env.set_dog_actions(dog_actions)
                        if self.train_base:
                            self._act_base_residual(inference=False)

                    residual_context = (
                        self.env.get_base_action_residual_obs().to(
                            self.device
                        )
                        if self.residual_context_shape is not None
                        else None
                    )
                    actions = self.alg.act(
                        obs,
                        critic_obs,
                        residual_context=residual_context,
                    )

                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # 多评论家：送进 storage 的是分组奖励向量；rewards 仍作标量用于日志
                    step_rewards = (self.env.rew_buf_vec.to(self.device)
                                    if self.alg.value_dim > 1 else rewards)

                    n, d = obs.size()
                    obs = self.obs_normalizer(obs.reshape(-1, self.env.num_one_step_obs)).reshape(n, d)

                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(step_rewards, dones, infos, next_critic_obs)

                    if self.log_dir is not None:
                        episode_info = infos.get('episode')
                        if (
                            episode_info
                            and int(episode_info.get('_episode_count', 0)) > 0
                        ):
                            ep_infos.append(episode_info)
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_estimation_loss,
                mean_gate_loss,
            ) = self.alg.update_simple()


            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            self.current_learning_iteration = it + 1
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        f'model_{self.current_learning_iteration}.pt',
                    )
                )
            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    @staticmethod
    def _aggregate_episode_infos(ep_infos):
        """Weight batch means by the number of episodes that produced them."""
        if not ep_infos:
            return {}
        keys = sorted(
            {
                key
                for info in ep_infos
                for key in info
                if not key.startswith("_")
            }
        )
        metrics = {}
        for key in keys:
            weighted_sum = None
            total_weight = 0
            for info in ep_infos:
                if key not in info:
                    continue
                count = int(info.get("_episode_count", 1))
                value = torch.as_tensor(
                    info[key], dtype=torch.float32
                ).reshape(-1)
                contribution = value.sum() * count
                weighted_sum = (
                    contribution
                    if weighted_sum is None
                    else weighted_sum + contribution
                )
                total_weight += count
            if weighted_sum is not None and total_weight > 0:
                metrics[key] = weighted_sum / total_weight
        return metrics

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key, value in self._aggregate_episode_infos(
                locs['ep_infos']
            ).items():
                value = value.to(self.device)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        std_values = self.alg.actor_critic.std.detach()
        mean_std = std_values.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Payload', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/LoadGate', locs['mean_gate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/entropy_coef', self.alg.entropy_coef, locs['it'])
        update_stats = getattr(self.alg, "last_update_stats", {})
        if locs['it'] == 0:
            self.writer.add_text(
                "Model/pretrained_initialization",
                (
                    f"checkpoint={getattr(self.alg.actor_critic, '_pretrained_init_checkpoint', None)} "
                    f"keys={getattr(self.alg.actor_critic, '_pretrained_initialized_keys', None)} "
                    f"finetune_mode={getattr(self.alg.actor_critic, 'finetune_mode', None)}"
                ),
                0,
            )
        mapping = {
            "critic_value_loss_0": "Critic/task_value_loss",
            "critic_value_loss_1": "Critic/stability_value_loss",
            "critic_value_loss_2": "Critic/energy_value_loss",
            "critic_return_0": "Critic/task_return",
            "critic_return_1": "Critic/stability_return",
            "critic_return_2": "Critic/energy_return",
            "critic_advantage_std_0": "Critic/task_advantage_std",
            "critic_advantage_std_1": "Critic/stability_advantage_std",
            "critic_advantage_std_2": "Critic/energy_advantage_std",
            "raw_advantage_std_0": "Critic/task_raw_advantage_std",
            "raw_advantage_std_1": "Critic/stability_raw_advantage_std",
            "raw_advantage_std_2": "Critic/energy_raw_advantage_std",
            "fused_advantage_std": "Fusion/fused_advantage_std",
            "fused_advantage_component_rms_0": "Fusion/task_component_rms",
            "fused_advantage_component_rms_1": "Fusion/stability_component_rms",
            "fused_advantage_component_rms_2": "Fusion/energy_component_rms",
            "payload_position_mse": "Payload/position_mse",
            "payload_torque_mse": "Payload/gravity_torque_mse",
            "payload_velocity_mse": "Payload/com_velocity_mse",
            "payload_total_mse": "Payload/total_mse",
            "risk_risk": "Payload/risk_score",
            "risk_position": "Payload/com_position_norm",
            "risk_torque": "Payload/gravity_torque_norm",
            "risk_velocity": "Payload/com_velocity_norm",
            "weight_entropy": "LoadGate/weight_entropy",
            "weight_change_norm": "LoadGate/weight_change_norm",
            "weight_kl": "LoadGate/weight_kl",
            "weight_smooth_loss": "LoadGate/weight_smooth_loss",
            "ppo_grad_norm": "Loss/ppo_grad_norm",
            "payload_grad_norm": "Loss/payload_grad_norm",
            "gate_grad_norm": "Loss/gate_grad_norm",
            "total_fused_return": "Critic/total_fused_return",
            "expert_gate_entropy": "Gate/expert_gate_entropy",
            "expert_gate_load_balance": "Gate/expert_load_balance",
            "actor_entropy": "Policy/actor_entropy",
            "actor_kl": "Policy/actor_kl",
            "std_clamp_fraction": "Policy/std_clamp_fraction",
        }
        for source, destination in mapping.items():
            if source in update_stats:
                self.writer.add_scalar(
                    destination, update_stats[source], locs['it']
                )
        for weight_name in ("task", "stability", "energy"):
            for kind in ("predicted", "prior", "actual"):
                key = f"{kind}_{weight_name}_weight"
                if key in update_stats:
                    self.writer.add_scalar(
                        f"LoadGate/{kind}_{weight_name}",
                        update_stats[key],
                        locs['it'],
                    )
                std_key = f"{kind}_{weight_name}_weight_std"
                if std_key in update_stats:
                    self.writer.add_scalar(
                        f"LoadGate/{kind}_{weight_name}_std",
                        update_stats[std_key],
                        locs['it'],
                    )
            actual_key = f"actual_{weight_name}_weight"
            actual_std_key = f"actual_{weight_name}_weight_std"
            if actual_key in update_stats:
                self.writer.add_scalar(
                    f"LoadGate/{weight_name}_weight",
                    update_stats[actual_key],
                    locs['it'],
                )
            if actual_std_key in update_stats:
                self.writer.add_scalar(
                    f"LoadGate/{weight_name}_weight_std",
                    update_stats[actual_std_key],
                    locs['it'],
                )
        for key, value in update_stats.items():
            if key.startswith("expert_gate_weight_"):
                mode = key[len("expert_gate_weight_"):]
                if mode.isdigit():
                    self.writer.add_scalar(
                        f"Gate/expert_weight_{mode}",
                        value,
                        locs['it'],
                    )
                elif mode.startswith("std_"):
                    expert = mode[len("std_"):]
                    self.writer.add_scalar(
                        f"Gate/expert_weight_std_{expert}",
                        value,
                        locs['it'],
                    )
            elif key.startswith("payload_predicted_"):
                suffix = key[len("payload_predicted_"):]
                if suffix.isdigit():
                    self.writer.add_scalar(
                        f"Payload/estimated_state_{suffix}",
                        value,
                        locs['it'],
                    )
                else:
                    self.writer.add_scalar(
                        f"Payload/estimated_{suffix}",
                        value,
                        locs['it'],
                    )
            elif key.startswith("payload_true_"):
                suffix = key[len("payload_true_"):]
                if suffix.isdigit():
                    self.writer.add_scalar(
                        f"Payload/true_state_{suffix}",
                        value,
                        locs['it'],
                    )
                else:
                    self.writer.add_scalar(
                        f"Payload/true_{suffix}",
                        value,
                        locs['it'],
                    )
        # 底座顺应残差监控：残差幅度 + 底座姿态（看底座有没有为臂下蹲/前倾）
        if self.train_base:
            self.writer.add_scalar('BaseResidual/mean_abs', self.env.dog_residual.abs().mean().item(), locs['it'])
            self.writer.add_scalar('BaseResidual/max_abs', self.env.dog_residual.abs().max().item(), locs['it'])
            self.writer.add_scalar('BaseResidual/std', self.alg.base_residual.std.mean().item(), locs['it'])
            self.writer.add_scalar('BaseResidual/entropy_coef', self.alg.base_entropy_coef, locs['it'])
            self.writer.add_scalar('BaseResidual/base_height', self.env.root_states[:, 2].mean().item(), locs['it'])
            if hasattr(self.env, 'pitch'):
                self.writer.add_scalar('BaseResidual/base_pitch_abs', self.env.pitch.abs().mean().item(), locs['it'])
        action_residual = getattr(
            self.alg.actor_critic, "last_action_residual", None
        )
        if action_residual is not None:
            self.writer.add_scalar(
                "BaseActionResidual/mean_abs",
                action_residual.abs().mean().item(),
                locs["it"],
            )
            self.writer.add_scalar(
                "BaseActionResidual/max_abs",
                action_residual.abs().max().item(),
                locs["it"],
            )
            self.writer.add_scalar(
                "BaseActionResidual/saturation_fraction",
                (
                    action_residual.abs()
                    >= 0.99
                    * self.alg.actor_critic.base_action_residual_clip
                )
                .float()
                .mean()
                .item(),
                locs["it"],
            )
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar(
            'Policy/min_noise_std', std_values.min().item(), locs['it']
        )
        self.writer.add_scalar(
            'Policy/max_noise_std', std_values.max().item(), locs['it']
        )
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Payload / gate loss:':>{pad}} {locs['mean_estimation_loss']:.6f} / {locs['mean_gate_loss']:.6f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Payload / gate loss:':>{pad}} {locs['mean_estimation_loss']:.6f} / {locs['mean_gate_loss']:.6f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['tot_iter'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        save_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'payload_optimizer_state_dict': (
                self.alg.payload_optimizer.state_dict()
                if self.alg.payload_optimizer is not None
                else None
            ),
            'load_gate_optimizer_state_dict': (
                self.alg.load_gate_optimizer.state_dict()
                if self.alg.load_gate_optimizer is not None
                else None
            ),
            'architecture_metadata': (
                self.alg.actor_critic.architecture_metadata()
                if hasattr(self.alg.actor_critic, "architecture_metadata")
                else None
            ),
            'pretrained_initialization': {
                "checkpoint": getattr(
                    self.alg.actor_critic,
                    "_pretrained_init_checkpoint",
                    None,
                ),
                "initialized_keys": getattr(
                    self.alg.actor_critic,
                    "_pretrained_initialized_keys",
                    None,
                ),
                "architecture_version": getattr(
                    self.alg.actor_critic,
                    "_pretrained_architecture_version",
                    None,
                ),
            },
            'env_curriculum_state': (
                self.env.get_curriculum_state()
                if hasattr(self.env, "get_curriculum_state")
                else None
            ),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.empirical_normalization:
            save_dict['obs_norm_state_dict'] = self.obs_normalizer.state_dict()
            save_dict['critic_obs_norm_state_dict'] = self.critic_obs_normalizer.state_dict()
        # 底座顺应残差（联合 fine-tune）
        if self.train_base:
            save_dict['base_residual_state_dict'] = self.base_residual.state_dict()
            save_dict['base_optimizer_state_dict'] = self.alg.base_optimizer.state_dict()
        torch.save(save_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        expected_metadata = (
            self.alg.actor_critic.architecture_metadata()
            if hasattr(self.alg.actor_critic, "architecture_metadata")
            else None
        )
        saved_metadata = loaded_dict.get("architecture_metadata")
        if (
            expected_metadata is not None
            and saved_metadata is not None
            and saved_metadata != expected_metadata
        ):
            raise RuntimeError(
                "Checkpoint architecture metadata mismatch: "
                f"expected {expected_metadata}, got {saved_metadata}"
            )
        try:
            self.alg.actor_critic.load_state_dict(
                loaded_dict["model_state_dict"]
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Actor-critic checkpoint is incompatible. Single-critic and "
                "multi-critic models are intentionally not auto-converted. "
                f"Original error: {error}"
            ) from error
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if (
                self.alg.payload_optimizer is not None
                and loaded_dict.get("payload_optimizer_state_dict") is not None
            ):
                self.alg.payload_optimizer.load_state_dict(
                    loaded_dict["payload_optimizer_state_dict"]
                )
            if (
                self.alg.load_gate_optimizer is not None
                and loaded_dict.get("load_gate_optimizer_state_dict") is not None
            ):
                self.alg.load_gate_optimizer.load_state_dict(
                    loaded_dict["load_gate_optimizer_state_dict"]
                )
        if self.empirical_normalization:
            # Rollout buffers may contain inference tensors under PyTorch 1.10.
            # Loading within inference mode permits their in-place copy.
            with torch.inference_mode():
                obs_norm_state = (
                    loaded_dict.get("obs_norm_state_dict")
                    or loaded_dict.get("obs_normalizer_state_dict")
                )
                if obs_norm_state is not None:
                    self.obs_normalizer.load_state_dict(
                        obs_norm_state
                    )
                else:
                    print(
                        "[Normalization] checkpoint has no observation "
                        "statistics; starting from identity"
                    )
                if (
                    loaded_dict.get("critic_obs_norm_state_dict")
                    is not None
                ):
                    self.critic_obs_normalizer.load_state_dict(
                        loaded_dict["critic_obs_norm_state_dict"]
                    )
                else:
                    print(
                        "[Normalization] checkpoint has no critic "
                        "statistics; starting from identity"
                    )
        if self.train_base and 'base_residual_state_dict' in loaded_dict:
            self.base_residual.load_state_dict(loaded_dict['base_residual_state_dict'])
            if load_optimizer and 'base_optimizer_state_dict' in loaded_dict:
                self.alg.base_optimizer.load_state_dict(loaded_dict['base_optimizer_state_dict'])
        if (
            loaded_dict.get("env_curriculum_state") is not None
            and hasattr(self.env, "load_curriculum_state")
        ):
            self.env.load_curriculum_state(
                loaded_dict["env_curriculum_state"]
            )
        if loaded_dict.get("pretrained_initialization"):
            initialization = loaded_dict["pretrained_initialization"]
            self.alg.actor_critic._pretrained_init_checkpoint = (
                initialization.get("checkpoint")
            )
            self.alg.actor_critic._pretrained_initialized_keys = (
                initialization.get("initialized_keys")
            )
            self.alg.actor_critic._pretrained_architecture_version = (
                initialization.get("architecture_version")
            )
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        if getattr(
            self.alg.actor_critic,
            "base_action_residual_enabled",
            False,
        ):
            return self._inference_policy_with_residual_context
        return self.alg.actor_critic.act_inference

    def _inference_policy_with_residual_context(self, observations):
        residual_context = self.env.get_base_action_residual_obs().to(
            self.device
        )
        return self.alg.actor_critic.act_inference(
            observations,
            residual_context=residual_context,
        )

    def _extract_model_state_dict(self, ckpt):
        if not isinstance(ckpt, dict):
            return ckpt
        for k in ["model_state_dict", "actor_critic_state_dict", "state_dict"]:
            if k in ckpt:
                return ckpt[k]
        return ckpt


    def _load_matching_state_dict(self, model, src_state_dict):
        dst_state_dict = model.state_dict()
        matched = {}
        skipped = []

        for k, v in src_state_dict.items():
            if k in dst_state_dict and dst_state_dict[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)

        model.load_state_dict(matched, strict=False)
        print(f"[BasePolicy] loaded {len(matched)} params, skipped {len(skipped)} params")
        if len(skipped) > 0:
            print(f"[BasePolicy] first 10 skipped keys: {skipped[:10]}")

        # 宽松加载会静默跳过形状不符的参数。critic 因特权观测维度不同被跳过是正常的
        # （冻结推理用不到），但 actor / prop_encoder 一旦被跳过就等于用随机权重跑，
        # 训练照跑却毫无意义——必须直接报错。
        critical = [k for k in skipped if k.startswith(("actor.", "prop_encoder."))]
        if critical:
            raise RuntimeError(
                "冻结底座策略的关键权重形状不符、已被跳过，通常是 payload_dim / "
                f"观测维度与训练时不一致：{critical[:10]}"
            )

    def _build_and_load_base_policy(self, train_cfg):
        base_policy_path = self.cfg["base_policy_path"]
        base_policy_cfg = self.cfg.get("base_policy", {})
        base_policy_class_name = self.cfg.get(
            "base_policy_class_name", "Actor_mulCritic_attention1"
        )
        base_obs_dim = int(
            self.cfg.get("base_policy_obs_dim", self.env.num_dog_one_step_obs)
        )
        history_len = int(
            self.cfg.get("base_policy_history_len", self.env.history_length)
        )
        base_actor_obs_dim = base_obs_dim * history_len
        base_critic_obs_dim = int(
            self.cfg.get("base_policy_num_privileged_obs", base_actor_obs_dim)
        )

        configure_base_obs = getattr(
            self.env, "configure_base_policy_observation", None
        )
        if configure_base_obs is not None:
            configure_base_obs(history_len, base_obs_dim)

        self.base_policy_is_terrain_moe = (
            base_policy_class_name == "TerrainMoeActorCritic"
        )
        if self.base_policy_is_terrain_moe:
            self.base_policy = TerrainMoeActorCritic(
                base_actor_obs_dim,
                base_critic_obs_dim,
                self.env.num_dog_actions,
                base_obs_dim,
                None,
                history_len,
                **base_policy_cfg
            ).to(self.device)
        else:
            # Legacy frozen policy retained for compatibility with the old
            # go2_arm checkpoints.
            self.base_policy = Actor_mulCritic_attention1(
                base_actor_obs_dim,
                base_actor_obs_dim,
                self.env.num_dog_actions,
                base_obs_dim,
                base_obs_dim,
                history_len,
                payload_dim=int(self.cfg.get("base_policy_payload_dim", 0)),
                **base_policy_cfg
            ).to(self.device)

        ckpt = torch.load(base_policy_path, map_location=self.device)
        model_state_dict = self._extract_model_state_dict(ckpt)
        if self.base_policy_is_terrain_moe:
            expected_version = self.cfg.get(
                "base_policy_architecture_version"
            )
            checkpoint_version = (
                ckpt.get("architecture_version")
                if isinstance(ckpt, dict)
                else None
            )
            if (
                expected_version is not None
                and checkpoint_version != expected_version
            ):
                raise RuntimeError(
                    "Terrain-MoE base checkpoint architecture mismatch: "
                    f"expected {expected_version}, got {checkpoint_version}"
                )
            self.base_policy.load_state_dict(model_state_dict, strict=True)
            print(
                "[BasePolicy] strictly loaded Terrain-MoE base "
                f"({checkpoint_version})"
            )
        else:
            self._load_matching_state_dict(self.base_policy, model_state_dict)

        self.base_policy.eval()
        for p in self.base_policy.parameters():
            p.requires_grad = False

        if self.cfg.get("base_empirical_normalization", False):
            self.base_obs_normalizer = EmpiricalHistoryNormalization(
                shape=[base_obs_dim], until=1.0e8
            ).to(self.device)

            if isinstance(ckpt, dict) and "obs_norm_state_dict" in ckpt:
                try:
                    self.base_obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
                    print("[BasePolicy] loaded obs normalizer state")
                except Exception as e:
                    print(f"[BasePolicy] failed to load obs normalizer: {e}")

            self.base_obs_normalizer.eval()
            for p in self.base_obs_normalizer.parameters():
                p.requires_grad = False

        print(
            "[BasePolicy] frozen base policy loaded from: "
            f"{base_policy_path} "
            f"(class={base_policy_class_name}, obs={base_actor_obs_dim})"
        )


    def _build_base_residual(self, jf):
        """构建底座顺应残差网络（zero-init，初始等价于纯冻结底座）。"""
        self.base_residual = BaseResidual(
            num_obs=self.env.num_base_residual_obs,
            num_actions=self.env.num_dog_actions,
            hidden_dims=tuple(getattr(jf, "hidden_dims", (256, 256))),
            init_noise_std=getattr(jf, "init_noise_std", 0.1),
            residual_clip=getattr(jf, "residual_clip", 0.3),
            std_max=getattr(jf, "std_max", 0.5),
        ).to(self.device)

        self.alg.set_base_residual(
            self.base_residual,
            learning_rate=getattr(jf, "residual_lr", 2e-4),
            entropy_coef=getattr(jf, "residual_entropy_coef", 0.005),
            entropy_coef_final=getattr(jf, "residual_entropy_coef_final", None),
            entropy_anneal_start_frac=getattr(jf, "residual_entropy_anneal_start_frac", 0.25),
        )
        print(f"[BaseResidual] trainable compliance residual built: "
              f"obs={self.env.num_base_residual_obs} act={self.env.num_dog_actions} "
              f"lr={getattr(jf, 'residual_lr', 2e-4)} clip={getattr(jf, 'residual_clip', 0.3)}")

    def _act_base_residual(self, inference=False):
        """采样/推理底座残差并写入环境。"""
        base_obs = self.env.get_base_residual_obs().to(self.device)
        if inference:
            with torch.inference_mode():
                residual = self.base_residual.act_inference(base_obs)
        else:
            residual = self.alg.act_base(base_obs)
        self.env.set_dog_residual(residual)

    def _infer_dog_actions(self):
        if self.base_policy is None:
            return torch.zeros(
                self.env.num_envs,
                self.env.num_dog_actions,
                device=self.device,
                dtype=torch.float
            )

        dog_obs = self.env.get_dog_observations().to(self.device)

        if self.cfg.get("base_empirical_normalization", False):
            n, d = dog_obs.shape
            dog_obs = self.base_obs_normalizer(
                dog_obs.reshape(-1, self.env.num_dog_one_step_obs)
            ).reshape(n, d)

        with torch.inference_mode():
            dog_actions = self.base_policy.act_inference(dog_obs)
            if isinstance(dog_actions, tuple):
                dog_actions = dog_actions[0]

        return dog_actions
