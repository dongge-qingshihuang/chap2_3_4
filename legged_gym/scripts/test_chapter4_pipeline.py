#!/usr/bin/env python3
"""Short end-to-end Chapter 4 control and PPO smoke test."""

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.

import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    args.task = "go2_arm"
    args.num_envs = 1
    args.sim_device = "cuda:0"
    args.rl_device = "cuda:0"
    args.headless = True

    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, log_root=None
    )
    if runner._pretrained_obs_normalizer_state is not None:
        assert runner.empirical_normalization
        assert runner.obs_normalizer.count.item() > 0
    assert env.num_actions == env.num_dog_actions == 12
    assert env.num_obs == 900
    assert env.num_privileged_obs == 251
    assert runner.alg.value_dim == env.value_dim
    assert set(env.reward_group_idx) == {0, 1, 2}
    assert torch.allclose(
        env.reward_group_weights,
        torch.tensor([0.7, 0.2, 0.1], device=env.device),
    )
    if env.cfg.rewards.moving_arm_completion_enabled:
        assert abs(
            env.reward_scales["task__moving_arm_completion"] - 0.2
        ) < 1e-6
        saved_state = {
            name: getattr(env, name).clone()
            for name in (
                "executed_commands_snapshot",
                "base_lin_vel",
                "reset_buf",
                "settle_buf",
                "arm_settle_in_tol",
                "moving_completion_eligible",
                "moving_completion_awarded",
                "moving_completion_hold_counter",
                "moving_completion_tracking_sum",
                "moving_completion_progress",
                "moving_completion_required",
                "moving_completion_pending_reward",
                "moving_completion_event_buf",
                "moving_completion_moving_event_buf",
                "moving_completion_static_event_buf",
                "moving_completion_event_progress_ratio",
                "moving_completion_event_mean_tracking",
                "episode_moving_completion_count",
                "episode_static_completion_count",
                "episode_moving_completion_bonus_sum",
            )
        }

        def reset_window():
            env._reset_moving_completion_window(
                torch.arange(env.num_envs, device=env.device)
            )
            env.moving_completion_eligible[:] = True
            env.executed_commands_snapshot.zero_()
            env.base_lin_vel.zero_()
            env.reset_buf.zero_()

        def feed_window(command_speed, actual_speed):
            env.executed_commands_snapshot[:, 0] = command_speed
            env.base_lin_vel[:, 0] = actual_speed
            for step in range(
                env.cfg.rewards.moving_arm_completion_hold_steps
            ):
                env.arm_settle_in_tol[:] = True
                env.settle_buf[:] = step + 1 == (
                    env.cfg.rewards.moving_arm_completion_hold_steps
                )
                env._update_moving_completion_window()

        reset_window()
        feed_window(0.5, 0.5)
        reward = env._reward_moving_arm_completion()
        assert torch.allclose(reward, torch.ones_like(reward), atol=1e-6)
        assert torch.equal(
            env._reward_moving_arm_completion(),
            torch.zeros_like(reward),
        )
        assert torch.allclose(
            env.episode_moving_completion_bonus_sum,
            torch.full_like(reward, 0.2),
            atol=1e-6,
        )

        reset_window()
        feed_window(0.5, -0.5)
        assert torch.equal(
            env._reward_moving_arm_completion(),
            torch.zeros(env.num_envs, device=env.device),
        )

        moving_count_before_static = (
            env.episode_moving_completion_count.clone()
        )
        reset_window()
        feed_window(0.0, 0.0)
        assert torch.equal(
            env._reward_moving_arm_completion(),
            torch.zeros(env.num_envs, device=env.device),
        )
        assert torch.all(env.episode_static_completion_count > 0)
        assert torch.equal(
            env.episode_moving_completion_count,
            moving_count_before_static,
        )

        reset_window()
        env.executed_commands_snapshot[:, 0] = 0.5
        env.base_lin_vel[:, 0] = 0.5
        for step in range(10):
            env.arm_settle_in_tol[:] = True
            env.settle_buf[:] = False
            env._update_moving_completion_window()
        env.arm_settle_in_tol[:] = False
        env._update_moving_completion_window()
        for step in range(
            env.cfg.rewards.moving_arm_completion_hold_steps - 1
        ):
            env.arm_settle_in_tol[:] = True
            env.settle_buf[:] = False
            env._update_moving_completion_window()
        assert torch.equal(
            env._reward_moving_arm_completion(),
            torch.zeros(env.num_envs, device=env.device),
        )

        reset_window()
        env.reset_buf[:] = True
        feed_window(0.5, 0.5)
        assert torch.equal(
            env._reward_moving_arm_completion(),
            torch.zeros(env.num_envs, device=env.device),
        )

        for name, value in saved_state.items():
            getattr(env, name).copy_(value)
    assert len(env.foot_names) == 4
    saved_feet_pos = env.feet_pos.clone()
    env.feet_pos.zero_()
    assert torch.isfinite(env._reward_foot_spacing()).all()
    assert (env._reward_foot_spacing() > 0).all()
    target_positions = {
        "FL": (0.30, 0.12),
        "FR": (0.30, -0.12),
        "RL": (-0.30, 0.12),
        "RR": (-0.30, -0.12),
    }
    for foot_index, foot_name in enumerate(env.foot_names):
        side = foot_name.split("_", 1)[0]
        env.feet_pos[:, foot_index, 0] = target_positions[side][0]
        env.feet_pos[:, foot_index, 1] = target_positions[side][1]
        env.feet_pos[:, foot_index, 2] = 0.0
    assert torch.allclose(
        env._reward_foot_spacing(),
        torch.zeros(env.num_envs, device=env.device),
        atol=1e-6,
    )
    env.feet_pos.copy_(saved_feet_pos)
    assert env.arm_fixed_warmup_iters == 1000
    env.ppo_it = env.arm_fixed_warmup_iters - 1
    assert env._arm_warmup_active()
    assert not env._arm_motion_enabled()
    env.ppo_it = env.arm_fixed_warmup_iters
    assert not env._arm_warmup_active()
    assert env._arm_motion_enabled()
    env.ppo_it = 0
    indexed_height = env._get_base_heights(
        torch.tensor([0], device=env.device)
    )
    assert indexed_height.numel() == 1

    # Pure timeout at exactly max_episode_length.
    env.episode_length_buf.fill_(env.max_episode_length)
    env.base_quat.zero_()
    env.base_quat[:, 3] = 1.0
    env.base_contact_counter.zero_()
    env.contact_forces.zero_()
    env.check_termination()
    assert env.time_out_buf.all()
    assert env.reset_buf.all()

    # Fall has priority over timeout on the same step.
    env.episode_length_buf.fill_(env.max_episode_length)
    env.base_quat.zero_()
    env.base_quat[:, 0] = 2 ** -0.5
    env.base_quat[:, 3] = 2 ** -0.5
    env.check_termination()
    assert env.reset_buf.all()
    assert not env.time_out_buf.any()

    # Sustained base contact terminates even when orientation is valid.
    env.episode_length_buf.zero_()
    env.base_quat.zero_()
    env.base_quat[:, 3] = 1.0
    env.contact_forces.zero_()
    env.base_contact_counter.fill_(
        env.cfg.termination.base_contact_persist_steps - 1
    )
    env.contact_forces[
        :, env.termination_contact_indices, 2
    ] = -env.cfg.termination.base_contact_force_threshold * 2.0
    env.check_termination()
    assert env.base_contact_termination_buf.all()
    assert env.reset_buf.all()
    assert not env.time_out_buf.any()
    assert abs(
        env.reward_scales["termination"] + 500.0 * env.dt
    ) < 1e-8
    assert abs(
        env.reward_scales["stab__base_contact"] + 5.0 * env.dt
    ) < 1e-8
    assert env._reward_termination().all()
    assert env._reward_base_contact().all()

    # Resetting an environment must prevent history leakage across episodes.
    for frame in env.dog_obs_history:
        frame[0] = 1.0
    env.reset_idx(torch.tensor([0], device=env.device))
    assert all(not frame[0].any() for frame in env.dog_obs_history)
    assert "episode" in env.extras
    for key in (
        "roll_termination_rate",
        "pitch_termination_rate",
        "fall_termination_rate",
        "timeout_rate",
        "arm_warmup_active",
    ):
        assert key in env.extras["episode"]
    env.reset_idx(torch.tensor([], dtype=torch.long, device=env.device))
    assert "episode" not in env.extras

    observations, critic_observations = env.reset()
    for step_index in range(runner.num_steps_per_env):
        residual_context = (
            env.get_base_action_residual_obs()
            if runner.alg.actor_critic.base_action_residual_enabled
            else None
        )
        actions = runner.alg.act(
            observations,
            critic_observations,
            residual_context=residual_context,
        )
        (
            observations,
            privileged,
            rewards,
            dones,
            infos,
            termination_ids,
            termination_privileged,
        ) = env.step(actions)
        assert "time_outs" in infos
        assert infos["time_outs"].shape == (env.num_envs,)
        critic_observations = privileged.to(env.device)
        rewards = rewards.to(env.device)
        dones = dones.to(env.device)
        next_critic = critic_observations.clone()
        next_critic[termination_ids.to(env.device)] = (
            termination_privileged.to(env.device)
        )
        runner.alg.process_env_step(
            env.rew_buf_vec.to(env.device),
            dones,
            infos,
            next_critic,
        )

    assert torch.isfinite(env.torques).all()
    assert (env.torques.abs() <= env.torque_limits + 1e-5).all()
    runner.alg.compute_returns(critic_observations)
    runner.alg.update_simple()
    stats = runner.alg.last_update_stats
    assert all(
        torch.isfinite(torch.tensor(stats[f"critic_value_loss_{index}"]))
        for index in range(env.value_dim)
    )
    if env.value_dim > 1:
        assert all(
            torch.isfinite(torch.tensor(stats[f"actual_{name}_weight"]))
            for name in ("task", "stability", "energy")
        )
    if runner.alg.actor_critic.base_action_residual_enabled:
        assert runner.alg.actor_critic.actor[0].weight.grad is None
        assert runner.alg.actor_critic.control_gate[0].weight.grad is None
        assert (
            runner.alg.actor_critic.control_adapters[0][0].weight.grad
            is None
        )
        assert any(
            parameter.grad is not None
            for parameter in (
                runner.alg.actor_critic.base_action_residual.parameters()
            )
        )
    else:
        assert runner.alg.actor_critic.actor[0].weight.grad is not None
        assert runner.alg.actor_critic.control_gate[0].weight.grad is not None
        assert (
            runner.alg.actor_critic.control_adapters[0][0].weight.grad
            is not None
        )
    assert all(
        parameter.grad is None
        for parameter in runner.alg.actor_critic.TsDyn_modules.parameters()
    )

    curriculum_state = env.get_curriculum_state()
    assert curriculum_state is not None
    assert "stage" in curriculum_state
    env.gym.destroy_sim(env.sim)
    print("chapter4_pipeline_smoke_test OK")


if __name__ == "__main__":
    main()
