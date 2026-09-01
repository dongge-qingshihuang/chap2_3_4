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

        self.policy_class_name = self.cfg["policy_class_name"]
        actor_critic_class = eval(self.policy_class_name)

        # 这一轮先保持与当前 env 接口一致
        # 后面 env 改成 arm policy 真控机械臂后，再切到 self.env.num_arm_actions
        self.train_action_dim = self.env.num_actions

        if self.policy_class_name == "ActorCritic":
            actor_critic = actor_critic_class(
                self.env.num_obs,
                num_critic_obs,
                self.train_action_dim,
                **self.policy_cfg
            ).to(self.device)
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

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.train_action_dim],
            base_obs_shape=base_obs_shape,
            base_actions_shape=base_actions_shape,
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

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

                    actions = self.alg.act(obs, critic_obs)

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
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
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

            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.alg.update_simple()


            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/entropy_coef', self.alg.entropy_coef, locs['it'])
        # 底座顺应残差监控：残差幅度 + 底座姿态（看底座有没有为臂下蹲/前倾）
        if self.train_base:
            self.writer.add_scalar('BaseResidual/mean_abs', self.env.dog_residual.abs().mean().item(), locs['it'])
            self.writer.add_scalar('BaseResidual/max_abs', self.env.dog_residual.abs().max().item(), locs['it'])
            self.writer.add_scalar('BaseResidual/std', self.alg.base_residual.std.mean().item(), locs['it'])
            self.writer.add_scalar('BaseResidual/entropy_coef', self.alg.base_entropy_coef, locs['it'])
            self.writer.add_scalar('BaseResidual/base_height', self.env.root_states[:, 2].mean().item(), locs['it'])
            if hasattr(self.env, 'pitch'):
                self.writer.add_scalar('BaseResidual/base_pitch_abs', self.env.pitch.abs().mean().item(), locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
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
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        save_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
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
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            # self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if self.train_base and 'base_residual_state_dict' in loaded_dict:
            self.base_residual.load_state_dict(loaded_dict['base_residual_state_dict'])
            if load_optimizer and 'base_optimizer_state_dict' in loaded_dict:
                self.alg.base_optimizer.load_state_dict(loaded_dict['base_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

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
        base_policy_cfg = train_cfg.get("base_policy", {})

        dog_obs_dim = self.env.num_dog_one_step_obs * self.env.history_length
        dog_critic_obs_dim = dog_obs_dim

        # 冻结策略的结构由它自己的 checkpoint 决定，而不是当前 env 配置：
        # 用第二章训练该 checkpoint 时的 payload_dim，不一致会导致 actor 首层形状不符
        self.base_policy = Actor_mulCritic_attention1(
            dog_obs_dim,
            dog_critic_obs_dim,
            self.env.num_dog_actions,
            self.env.num_dog_one_step_obs,
            self.env.num_dog_one_step_obs,
            self.env.history_length,
            payload_dim=int(self.cfg.get("base_policy_payload_dim", 0)),
            **base_policy_cfg
        ).to(self.device)

        ckpt = torch.load(base_policy_path, map_location=self.device)
        model_state_dict = self._extract_model_state_dict(ckpt)
        self._load_matching_state_dict(self.base_policy, model_state_dict)

        self.base_policy.eval()
        for p in self.base_policy.parameters():
            p.requires_grad = False

        if self.cfg.get("base_empirical_normalization", False):
            self.base_obs_normalizer = EmpiricalHistoryNormalization(
                shape=[self.env.num_dog_one_step_obs], until=1.0e8
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

        print(f"[BasePolicy] frozen base policy loaded from: {base_policy_path}")


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
            dog_actions = self.base_policy.act_inference(dog_obs)[0]

        return dog_actions