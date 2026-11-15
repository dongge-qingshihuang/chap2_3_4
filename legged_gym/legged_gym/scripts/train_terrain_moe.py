import sys

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def train(args):
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
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
    task_was_specified = any(
        argument == "--task" or argument.startswith("--task=")
        for argument in sys.argv[1:]
    )
    args = get_args()
    if not task_was_specified:
        args.task = "go2_terrain_moe_tcn_gate"
    args.headless = True
    train(args)
