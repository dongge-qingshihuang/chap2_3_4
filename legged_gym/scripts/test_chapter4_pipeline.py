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
    assert env.num_actions == env.num_dog_actions == 12
    assert env.num_obs == 900
    assert env.num_privileged_obs == 251
    assert runner.alg.value_dim == env.value_dim
    assert set(env.reward_group_idx) == {0, 1, 2}
    assert torch.allclose(
        env.reward_group_weights,
        torch.tensor([0.7, 0.2, 0.1], device=env.device),
    )
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
        actions = runner.alg.act(observations, critic_observations)
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
