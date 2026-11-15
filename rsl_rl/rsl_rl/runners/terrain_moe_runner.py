import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import TerrainMoePPO
from rsl_rl.modules import (
    TerrainMoeActorCritic,
    TerrainMoeAttentionActorCritic,
    TerrainActorMoEActorCritic,
)


class TerrainMoeRunner:
    """Trainer and inference adapter for the Go2 terrain MoE task."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        policy_classes = {
            "TerrainMoeActorCritic": TerrainMoeActorCritic,
            "TerrainMoeAttentionActorCritic": (
                TerrainMoeAttentionActorCritic
            ),
            "TerrainActorMoEActorCritic": TerrainActorMoEActorCritic,
        }
        policy_class_name = self.cfg["policy_class_name"]
        if policy_class_name not in policy_classes:
            raise ValueError(
                f"Unknown terrain MoE policy class: {policy_class_name}"
            )
        self.actor_critic = policy_classes[policy_class_name](
            env.num_obs,
            env.num_privileged_obs,
            env.num_actions,
            env.num_one_step_obs,
            env.num_rec,
            env.num_obs_lens,
            **self.policy_cfg,
        ).to(device)
        self.alg = TerrainMoePPO(
            self.actor_critic, device=device, **self.alg_cfg
        )
        self.alg.init_storage(
            env.num_envs,
            self.cfg["num_steps_per_env"],
            [env.num_obs],
            [env.num_privileged_obs],
            [env.num_actions],
            self.policy_cfg["dynamics_target_dim"],
            self.policy_cfg["contact_dim"],
            self.policy_cfg["terrain_dim"],
            self.policy_cfg["future_contact_steps"],
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )
        self.alg.train_mode()

        observations = self.env.get_observations().to(self.device)
        critic_observations = self.env.get_privileged_observations().to(
            self.device
        )
        reward_buffer = deque(maxlen=100)
        length_buffer = deque(maxlen=100)
        episode_reward = torch.zeros(
            self.env.num_envs, device=self.device
        )
        episode_length = torch.zeros(
            self.env.num_envs, device=self.device
        )
        episode_infos = []

        total_iterations = self.current_learning_iteration + num_learning_iterations
        for iteration in range(
            self.current_learning_iteration, total_iterations
        ):
            start_time = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(
                        observations, critic_observations
                    )
                    (
                        observations,
                        privileged_observations,
                        rewards,
                        dones,
                        infos,
                        _,
                        _,
                    ) = self.env.step(actions)
                    observations = observations.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    if privileged_observations is None:
                        critic_observations = observations
                    else:
                        critic_observations = privileged_observations.to(
                            self.device
                        )
                    self.alg.process_env_step(
                        rewards, dones, infos, self.env.moe_targets
                    )

                    if "episode" in infos:
                        episode_infos.append(infos["episode"])
                    episode_reward += rewards
                    episode_length += 1
                    finished = dones.nonzero(as_tuple=False).flatten()
                    if finished.numel() > 0:
                        reward_buffer.extend(
                            episode_reward[finished].cpu().tolist()
                        )
                        length_buffer.extend(
                            episode_length[finished].cpu().tolist()
                        )
                        episode_reward[finished] = 0.0
                        episode_length[finished] = 0.0

                collection_time = time.time() - start_time
                self.alg.compute_returns(critic_observations)

            update_start = time.time()
            stats = self.alg.update()
            learning_time = time.time() - update_start
            self.current_learning_iteration = iteration
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learning_time

            if self.log_dir is not None:
                self._log(
                    iteration,
                    stats,
                    episode_infos,
                    reward_buffer,
                    length_buffer,
                    collection_time,
                    learning_time,
                    total_iterations,
                )
                if iteration % self.save_interval == 0:
                    self.save(
                        os.path.join(
                            self.log_dir, f"model_{iteration}.pt"
                        )
                    )
            episode_infos.clear()

        if self.log_dir is not None:
            self.save(
                os.path.join(
                    self.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )

    def _log(
        self,
        iteration,
        stats,
        episode_infos,
        reward_buffer,
        length_buffer,
        collection_time,
        learning_time,
        total_iterations,
    ):
        episode_metrics = self._aggregate_episode_infos(episode_infos)
        tensorboard_keys = {
            "Loss/value_function": "value_loss",
            "Loss/surrogate": "surrogate_loss",
            "Loss/perception": "perception_loss",
            "Loss/policy": "policy_loss",
            "Loss/approx_KL": "approx_kl",
            "Loss/clip_fraction": "clip_fraction",
            "Loss/entropy": "entropy",
            "Loss/dynamics": "dynamics_loss",
            "Loss/contact": "contact_loss",
            "Loss/terrain": "terrain_loss",
            "Loss/KL": "kl_loss",
            "Loss/velocity": "velocity_loss",
            "Contact/valid_fraction": "contact_valid_fraction",
            "Gate/entropy": "gate_entropy",
            "Gate/temporal_norm": "gate_temporal_norm",
            "Gate/temporal_pre_norm": "gate_temporal_pre_norm",
            "Gate/weight_deviation_norm": "gate_weight_deviation_norm",
            "Gate/tcn_grad_norm": "gate_tcn_grad_norm",
            "Gate/balance_loss": "gate_balance_loss",
            "Perception/control_residual_norm": "residual_norm",
            "Perception/base_latent_norm": "base_latent_norm",
        }
        for tag, key in tensorboard_keys.items():
            self.writer.add_scalar(tag, stats[key], iteration)
        logged_stat_keys = set(tensorboard_keys.values())
        for key, value in stats.items():
            if key not in logged_stat_keys:
                if key.startswith("gate_weight_std_"):
                    tag = "Gate/std_" + key[len("gate_weight_std_") :]
                elif key.startswith("gate_weight_"):
                    tag = "Gate/weight_" + key[len("gate_weight_") :]
                elif key.startswith("gate_"):
                    tag = "Gate/" + key[len("gate_") :]
                else:
                    tag = "Loss/" + key
                self.writer.add_scalar(tag, value, iteration)
        mean_std = self.actor_critic.std.mean()
        self.writer.add_scalar(
            "Policy/mean_noise_std", mean_std.item(), iteration
        )
        if reward_buffer:
            self.writer.add_scalar(
                "Train/mean_reward",
                statistics.mean(reward_buffer),
                iteration,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(length_buffer),
                iteration,
            )
        for key, value in episode_metrics.items():
            self.writer.add_scalar(f"Episode/{key}", value, iteration)

        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / max(collection_time + learning_time, 1e-8)
        )
        self.writer.add_scalar("Perf/total_fps", fps, iteration)
        self.writer.add_scalar(
            "Perf/collection_time", collection_time, iteration
        )
        self.writer.add_scalar(
            "Perf/learning_time", learning_time, iteration
        )

        width = 80
        pad = 35
        iteration_time = collection_time + learning_time
        mean_reward = (
            statistics.mean(reward_buffer) if reward_buffer else float("nan")
        )
        mean_length = (
            statistics.mean(length_buffer) if length_buffer else float("nan")
        )
        if self.tot_time > 0:
            eta = (
                self.tot_time
                / max(iteration + 1, 1)
                * max(total_iterations - iteration - 1, 0)
            )
        else:
            eta = 0.0

        title = (
            f" Learning iteration {iteration}/{total_iterations} "
        )
        lines = [
            "#" * width,
            title.center(width),
            "",
            f"{'Computation:':>{pad}} {fps} steps/s "
            f"(collection: {collection_time:.3f}s, "
            f"learning: {learning_time:.3f}s)",
            f"{'Value function loss:':>{pad}} "
            f"{stats['value_loss']:.4f}",
            f"{'Surrogate loss:':>{pad}} "
            f"{stats['surrogate_loss']:.4f}",
            f"{'Policy / perception loss:':>{pad}} "
            f"{stats['policy_loss']:.4f} / "
            f"{stats['perception_loss']:.4f}",
            f"{'Approx KL / clip fraction:':>{pad}} "
            f"{stats['approx_kl']:.6f} / "
            f"{stats['clip_fraction']:.4f}",
            f"{'Dynamics / contact / terrain:':>{pad}} "
            f"{stats['dynamics_loss']:.4f} / "
            f"{stats['contact_loss']:.4f} / "
            f"{stats['terrain_loss']:.4f}",
            f"{'KL / velocity:':>{pad}} "
            f"{stats['kl_loss']:.4f} / "
            f"{stats['velocity_loss']:.4f}",
            f"{'Contact label validity:':>{pad}} "
            f"{stats['contact_valid_fraction']:.2%}",
            f"{'Gate entropy / residual norm:':>{pad}} "
            f"{stats['gate_entropy']:.4f} / "
            f"{stats['residual_norm']:.4f}",
            f"{'Gate type / temporal norm:':>{pad}} "
            f"{getattr(self.actor_critic, 'gate_type', 'attention')} / "
            f"{stats['gate_temporal_norm']:.4f}",
            f"{'Temporal feature pre/post norm:':>{pad}} "
            f"{stats['gate_temporal_pre_norm']:.4f} / "
            f"{stats['gate_temporal_norm']:.4f}",
            f"{'Gate batch-balance loss:':>{pad}} "
            f"{stats['gate_balance_loss']:.6f}",
            f"{'Gate weight deviation norm:':>{pad}} "
            f"{stats['gate_weight_deviation_norm']:.4f}",
            f"{'Gate TCN grad norm:':>{pad}} "
            f"{stats['gate_tcn_grad_norm']:.6f}",
            f"{'Mean action noise std:':>{pad}} "
            f"{mean_std.item():.4f}",
            f"{'Mean total reward:':>{pad}} {mean_reward:.3f}",
            f"{'Mean episode length:':>{pad}} {mean_length:.2f}",
        ]
        if "fused_dynamics_loss" in stats:
            lines.insert(
                10,
                f"{'Fused dynamics / contact / terrain:':>{pad}} "
                f"{stats['fused_dynamics_loss']:.4f} / "
                f"{stats['fused_contact_loss']:.4f} / "
                f"{stats['fused_terrain_loss']:.4f}",
            )
        gate_weights = [
            stats[key]
            for key in sorted(stats)
            if key.startswith("gate_weight_")
            and not key.startswith("gate_weight_std_")
            and key[len("gate_weight_") :].isdigit()
        ]
        gate_stds = [
            stats[key]
            for key in sorted(stats)
            if key.startswith("gate_weight_std_")
        ]
        if gate_weights:
            lines.insert(
                12,
                f"{'Mean gate weights:':>{pad}} "
                + " / ".join(f"{value:.4f}" for value in gate_weights),
            )
        if gate_stds:
            lines.insert(
                13,
                f"{'Gate weight std:':>{pad}} "
                + " / ".join(f"{value:.4f}" for value in gate_stds),
            )
        for key in sorted(episode_metrics):
            if key == "terrain_level":
                lines.append(
                    f"{'Mean terrain level:':>{pad}} "
                    f"{episode_metrics[key]:.4f}"
                )
            elif key.startswith("rew_"):
                lines.append(
                    f"{'Mean episode ' + key + ':':>{pad}} "
                    f"{episode_metrics[key]:.4f}"
                )
        lines.extend(
            [
                "-" * width,
                f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
                f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
                f"{'Total time:':>{pad}} {self.tot_time:.2f}s",
                f"{'ETA:':>{pad}} {eta:.1f}s",
                "-" * width,
            ]
        )
        print("\n".join(lines), flush=True)

    @staticmethod
    def _aggregate_episode_infos(episode_infos):
        if not episode_infos:
            return {}
        keys = sorted(
            {key for info in episode_infos for key in info.keys()}
        )
        metrics = {}
        for key in keys:
            values = [
                info[key]
                for info in episode_infos
                if key in info
            ]
            if not values:
                continue
            tensor_values = torch.cat(
                [
                    torch.as_tensor(
                        value, dtype=torch.float32
                    ).reshape(-1)
                    for value in values
                ]
            )
            if tensor_values.numel() > 0:
                metrics[key] = tensor_values.mean().item()
        return metrics

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "perception_optimizer_state_dict": (
                    self.alg.perception_optimizer.state_dict()
                    if self.alg.perception_optimizer is not None
                    else None
                ),
                "architecture_version": (
                    self.actor_critic.architecture_version
                ),
                "architecture_metadata": (
                    self.actor_critic.architecture_metadata()
                ),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        expected_version = self.actor_critic.architecture_version
        if checkpoint.get("architecture_version") != expected_version:
            raise RuntimeError(
                "Checkpoint architecture mismatch: "
                f"expected {expected_version}, "
                f"got {checkpoint.get('architecture_version')}"
            )
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            if (
                self.alg.perception_optimizer is not None
                and checkpoint.get("perception_optimizer_state_dict")
                is not None
            ):
                self.alg.perception_optimizer.load_state_dict(
                    checkpoint["perception_optimizer_state_dict"]
                )
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        if device is not None:
            self.actor_critic.to(device)
        self.actor_critic.eval()
        return self.actor_critic.act_inference
