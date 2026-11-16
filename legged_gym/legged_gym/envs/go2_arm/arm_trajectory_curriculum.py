"""Four-stage arm trajectory curriculum for load-aware Go2 training."""

import torch


class ArmTrajectoryCurriculum:
    STAGES = (
        {
            "name": "local",
            "delta_scale": 0.05,
            "interp_alpha": 0.0,
            "perturb_scale": 0.25,
            "current_mix": 0.9,
        },
        {
            "name": "medium",
            "delta_scale": 0.12,
            "interp_alpha": 0.2,
            "perturb_scale": 0.5,
            "current_mix": 0.6,
        },
        {
            "name": "large",
            "delta_scale": 0.25,
            "interp_alpha": 0.4,
            "perturb_scale": 0.8,
            "current_mix": 0.25,
        },
        {
            "name": "full",
            "delta_scale": 0.5,
            "interp_alpha": 1.0,
            "perturb_scale": 1.0,
            "current_mix": 0.0,
        },
    )

    def __init__(
        self,
        num_envs,
        device,
        enabled=True,
        min_iterations=(0, 200, 500, 1000),
        min_success_rate=0.6,
        min_stability_rate=0.6,
        evaluation_window=20,
        max_stage=3,
        load_position_coef=1.0,
        load_torque_coef=1.0,
        load_torque_rate_coef=1.0,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.enabled = bool(enabled)
        self.min_iterations = tuple(int(value) for value in min_iterations)
        self.min_success_rate = float(min_success_rate)
        self.min_stability_rate = float(min_stability_rate)
        self.evaluation_window = int(evaluation_window)
        self.max_stage = min(int(max_stage), len(self.STAGES) - 1)
        self.load_position_coef = float(load_position_coef)
        self.load_torque_coef = float(load_torque_coef)
        self.load_torque_rate_coef = float(load_torque_rate_coef)

        self.stage = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.attempts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.successes = torch.zeros_like(self.attempts)
        self.stable_attempts = torch.zeros_like(self.attempts)
        self.load_difficulty = torch.zeros(
            self.num_envs, device=self.device
        )
        self.previous_payload = torch.zeros(
            self.num_envs, 9, device=self.device
        )
        self.has_previous_payload = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.iteration = 0
        self.sampled_stage_counts = torch.zeros(
            len(self.STAGES), dtype=torch.long, device=self.device
        )
        self.ik_attempts = 0
        self.ik_successes = 0

    def begin_trajectories(self, env_ids):
        if not self.enabled or len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        counts = torch.bincount(
            self.stage[ids], minlength=len(self.STAGES)
        )
        self.sampled_stage_counts += counts
        self.attempts[ids] += 1

    def observe_payload(self, payload_state, dt):
        if not self.enabled:
            return
        current = payload_state.detach()
        delta = current - self.previous_payload
        torque_rate = delta[:, 3:6] / max(float(dt), 1e-6)
        difficulty = (
            self.load_position_coef * delta[:, 0:3].norm(dim=-1)
            + self.load_torque_coef * delta[:, 3:6].norm(dim=-1)
            + self.load_torque_rate_coef * torque_rate.norm(dim=-1)
        )
        self.load_difficulty = torch.where(
            self.has_previous_payload,
            difficulty,
            torch.zeros_like(difficulty),
        )
        self.previous_payload[:] = current
        self.has_previous_payload[:] = True

    def record_result(self, env_ids, success, stable):
        if not self.enabled or len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        success = torch.as_tensor(
            success, device=self.device, dtype=torch.bool
        ).reshape(-1)
        stable = torch.as_tensor(
            stable, device=self.device, dtype=torch.bool
        ).reshape(-1)
        active = self.attempts[ids] > 0
        ids = ids[active]
        success = success[active]
        stable = stable[active]
        if len(ids) == 0:
            return
        self.successes[ids] += success.long()
        self.stable_attempts[ids] += stable.long()

    def _sample_ik_based(self, env, env_ids, stage_index):
        stage = self.STAGES[stage_index]
        current = env.dof_pos[env_ids, env.arm_control_dof_slice]
        count = len(env_ids)
        lower = env.arm_q_lower.unsqueeze(0)
        upper = env.arm_q_upper.unsqueeze(0)
        if env.ik_lib_q is None:
            sampled = current + torch.randn_like(current) * stage["delta_scale"]
        else:
            first = torch.randint(
                0, env.ik_lib_size, (count,), device=self.device
            )
            second = torch.randint(
                0, env.ik_lib_size, (count,), device=self.device
            )
            alpha = torch.rand(count, 1, device=self.device) * stage[
                "interp_alpha"
            ]
            sampled = (
                (1.0 - alpha) * env.ik_lib_q[first]
                + alpha * env.ik_lib_q[second]
            )
            if isinstance(env.ik_perturb_std, torch.Tensor):
                sampled = sampled + env.ik_perturb_std * (
                    stage["perturb_scale"] * torch.randn_like(sampled)
                )
            else:
                sampled = sampled + float(env.ik_perturb_std) * (
                    stage["perturb_scale"] * torch.randn_like(sampled)
                )
        sampled = torch.clamp(sampled, lower, upper)
        mix = float(stage["current_mix"])
        sampled = mix * current + (1.0 - mix) * sampled
        if stage_index == 0:
            sampled = current + stage["delta_scale"] * torch.randn_like(
                current
            )
        return torch.clamp(sampled, lower, upper)

    def sample_q(self, env, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        sampled = env.dof_pos[ids, env.arm_control_dof_slice].clone()
        for stage_index in range(self.max_stage + 1):
            mask = self.stage[ids] == stage_index
            if not mask.any():
                continue
            candidate = self._sample_ik_based(
                env, ids[mask], stage_index
            )
            sampled[mask] = candidate
        self.ik_attempts += len(ids)

        lower = env.arm_q_lower.unsqueeze(0)
        upper = env.arm_q_upper.unsqueeze(0)
        for _ in range(8):
            ee_pos, _ = env._fk_arm_mount_to_ee(sampled)
            reach = ee_pos.norm(dim=-1)
            bad = (
                (reach < env.joint_sample_l_min)
                | (reach > env.joint_sample_l_max)
                | env._ee_in_body(ee_pos)
            )
            if not bad.any():
                break
            for stage_index in range(self.max_stage + 1):
                stage_bad = bad & (self.stage[ids] == stage_index)
                if stage_bad.any():
                    sampled[stage_bad] = self._sample_ik_based(
                        env,
                        ids[stage_bad],
                        stage_index,
                    )
        final_ee, _ = env._fk_arm_mount_to_ee(sampled)
        final_reach = final_ee.norm(dim=-1)
        valid = (
            (final_reach >= env.joint_sample_l_min)
            & (final_reach <= env.joint_sample_l_max)
            & ~env._ee_in_body(final_ee)
        )
        self.ik_successes += int(valid.sum().item())
        fallback = env.dof_pos[ids, env.arm_control_dof_slice]
        sampled[~valid] = fallback[~valid]
        return torch.clamp(sampled, lower, upper)

    def update_stages(self, iteration):
        if not self.enabled:
            return
        self.iteration = int(iteration)
        for stage_index in range(self.max_stage):
            next_stage = stage_index + 1
            eligible = (
                (self.stage == stage_index)
                & (self.attempts >= self.evaluation_window)
                & (self.iteration >= self.min_iterations[next_stage])
            )
            if not eligible.any():
                continue
            success_rate = (
                self.successes.float() / self.attempts.clamp_min(1).float()
            )
            stability_rate = (
                self.stable_attempts.float()
                / self.attempts.clamp_min(1).float()
            )
            promote = (
                eligible
                & (success_rate >= self.min_success_rate)
                & (stability_rate >= self.min_stability_rate)
            )
            self.stage[promote] = next_stage
            self.attempts[promote] = 0
            self.successes[promote] = 0
            self.stable_attempts[promote] = 0

    def metadata(self):
        denom = self.attempts.clamp_min(1).float()
        stage_counts = torch.bincount(
            self.stage, minlength=len(self.STAGES)
        ).float()
        return {
            "stage": self.stage,
            "mean_success_rate": (
                self.successes.float() / denom
            ).mean(),
            "mean_stability_rate": (
                self.stable_attempts.float() / denom
            ).mean(),
            "mean_load_difficulty": self.load_difficulty.mean(),
            "stage_sampling": (
                self.sampled_stage_counts.float()
                / self.sampled_stage_counts.sum().clamp_min(1)
            ),
            "stage_counts": stage_counts / max(self.num_envs, 1),
            "ik_success_rate": self.ik_successes
            / max(self.ik_attempts, 1),
        }

    def state_dict(self):
        return {
            "stage": self.stage.cpu(),
            "attempts": self.attempts.cpu(),
            "successes": self.successes.cpu(),
            "stable_attempts": self.stable_attempts.cpu(),
            "load_difficulty": self.load_difficulty.cpu(),
            "previous_payload": self.previous_payload.cpu(),
            "has_previous_payload": self.has_previous_payload.cpu(),
            "iteration": self.iteration,
            "sampled_stage_counts": self.sampled_stage_counts.cpu(),
            "ik_attempts": self.ik_attempts,
            "ik_successes": self.ik_successes,
        }

    def load_state_dict(self, state):
        if not state:
            return
        for key in (
            "stage",
            "attempts",
            "successes",
            "stable_attempts",
            "load_difficulty",
            "previous_payload",
            "has_previous_payload",
            "sampled_stage_counts",
        ):
            setattr(
                self,
                key,
                state[key].to(self.device),
            )
        self.iteration = int(state.get("iteration", 0))
        self.ik_attempts = int(state.get("ik_attempts", 0))
        self.ik_successes = int(state.get("ik_successes", 0))
