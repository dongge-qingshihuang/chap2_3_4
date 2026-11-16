#!/usr/bin/env python3
"""Train one Chapter 4 ablation selected by GO2_ARM_EXPERIMENT."""

import os

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    mode = os.environ.get("GO2_ARM_EXPERIMENT", "B0")
    args = get_args()
    args.task = "go2_arm"
    if args.num_envs is None:
        args.num_envs = 4096
    args.headless = True
    args.experiment_name = os.environ.get(
        "CHAPTER4_EXPERIMENT_NAME", "chapter4_ablation"
    )
    args.run_name = os.environ.get(
        "CHAPTER4_RUN_NAME", f"{mode}_s0"
    )
    if os.environ.get("CHAPTER4_RESUME", "0") == "1":
        args.resume = True

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        log_root=os.environ.get("CHAPTER4_LOG_ROOT", "default"),
    )
    print(
        "[Ablation] "
        f"mode={mode} "
        f"multi_critic={env_cfg.env.value_dim == 3} "
        f"curriculum={env_cfg.arm.trajectory_curriculum.enabled} "
        f"payload_aware={env_cfg.arm.payload.mode != 'none'} "
        f"load_aware={train_cfg.policy.load_aware_enabled} "
        f"pretrained={bool(train_cfg.runner.pretrained_base_checkpoint)} "
        f"run={args.experiment_name}/{args.run_name}",
        flush=True,
    )
    remaining_iterations = max(
        0,
        train_cfg.runner.max_iterations
        - runner.current_learning_iteration,
    )
    runner.learn(
        num_learning_iterations=remaining_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    main()
