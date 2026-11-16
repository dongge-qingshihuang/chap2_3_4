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
    assert runner.alg.value_dim == 3
    indexed_height = env._get_base_heights(
        torch.tensor([0], device=env.device)
    )
    assert indexed_height.numel() == 1

    # Resetting an environment must prevent history leakage across episodes.
    for frame in env.dog_obs_history:
        frame[0] = 1.0
    env.reset_idx(torch.tensor([0], device=env.device))
    assert all(not frame[0].any() for frame in env.dog_obs_history)

    observations, critic_observations = env.reset()
    for _ in range(runner.num_steps_per_env):
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
        for index in range(3)
    )
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
