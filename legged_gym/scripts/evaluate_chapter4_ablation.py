#!/usr/bin/env python3
"""Deterministic Chapter 4 ablation evaluation on flat and rough terrain."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import torch
from isaacgym.torch_utils import quat_apply

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "legged_gym"
    / "logs"
    / "chapter4_ablation_formal_bounded"
    / "runs"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "legged_gym"
    / "logs"
    / "chapter4_evaluation"
    / "results.json"
)
MODES = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B3R",
    "B3M",
    "B3RM",
    "B3RMD",
    "B4R",
    "B4",
    "B0F",
)
TERRAINS = ("flat", "rough")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--fixed-arm", action="store_true")
    parser.add_argument("--obs-normalization", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--terrain", choices=TERRAINS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--joint-min-progress", type=float, default=0.8)
    parser.add_argument("--joint-min-tracking", type=float, default=0.7)
    return parser.parse_args()


def find_checkpoint(run_root, mode):
    matches = sorted(
        run_root.glob(f"*_{mode}_v2_s0/model_8000.pt")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one final checkpoint for {mode}, got {matches}"
        )
    return matches[0]


def make_isaac_args(args):
    saved_argv = sys.argv
    try:
        sys.argv = [
            saved_argv[0],
            "--task",
            "go2_arm",
            "--num_envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--rl_device",
            args.device,
            "--headless",
        ]
        return get_args()
    finally:
        sys.argv = saved_argv


def configure_env_cfg(env_cfg, terrain):
    env_cfg.env.num_envs = env_cfg.env.num_envs
    env_cfg.noise.add_noise = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.terrain.flat_warmup_iters = 0
    # Build the curriculum terrain in both cases so the flat column can be
    # selected directly. Updates are disabled after environment creation.
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.mesh_type = "trimesh"
    for name in (
        "randomize_friction",
        "randomize_base_mass",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_link_mass",
        "randomize_restitution",
        "randomize_motor_strength",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "push_robots",
        "delay",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def prepare_rough_terrain(env):
    # Curriculum terrain columns cover flat, rough planes, slopes, stairs and
    # discrete obstacles. Exclude the flat and rough-flat columns and spread
    # environments over the remaining representative types at level 3.
    columns = torch.tensor(
        [2, 4, 6, 7, 8, 10, 12, 14, 16, 18, 19],
        device=env.device,
        dtype=torch.long,
    )
    env.terrain_types[:] = columns[
        torch.arange(env.num_envs, device=env.device) % len(columns)
    ]
    env.terrain_levels.fill_(3)
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types
    ]


def prepare_flat_terrain(env):
    env.terrain_types.fill_(0)
    env.terrain_levels.fill_(0)
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types
    ]


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sample_base_targets(env, env_ids, targets, speeds, radius_min=0.8, radius_max=2.2):
    if len(env_ids) == 0:
        return
    count = len(env_ids)
    origins = env.env_origins[env_ids, :2]
    angles = torch.rand(count, device=env.device) * (2.0 * math.pi)
    radii = radius_min + torch.rand(count, device=env.device) * (
        radius_max - radius_min
    )
    targets[env_ids, 0] = origins[:, 0] + radii * torch.cos(angles)
    targets[env_ids, 1] = origins[:, 1] + radii * torch.sin(angles)
    speeds[env_ids] = 0.5 + 0.4 * torch.rand(
        count, device=env.device
    )


def set_base_commands(env, targets, speeds):
    delta = targets - env.root_states[:, :2]
    desired_yaw = torch.atan2(delta[:, 1], delta[:, 0])
    forward = quat_apply(env.base_quat, env.forward_vec)
    current_yaw = torch.atan2(forward[:, 1], forward[:, 0])
    yaw_error = wrap_to_pi(desired_yaw - current_yaw)
    env.commands[:, 0] = speeds
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = torch.clamp(0.5 * yaw_error, -2.0, 2.0)
    env.commands[:, 3] = desired_yaw


def evaluate_single(args):
    if not args.single or args.mode is None or args.terrain is None:
        raise ValueError("--single requires --mode and --terrain")
    checkpoint = args.checkpoint or find_checkpoint(
        args.run_root, args.mode
    )
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    os.environ["GO2_ARM_EXPERIMENT"] = args.mode
    config_name = os.environ.get("GO2_ARM_EXPERIMENT", "B0")
    if config_name != args.mode:
        raise RuntimeError(
            f"Config mode mismatch: environment={config_name}, "
            f"requested={args.mode}"
        )

    env_cfg, train_cfg = task_registry.get_cfgs("go2_arm")
    configure_env_cfg(env_cfg, args.terrain)
    train_cfg.runner.empirical_normalization = bool(
        args.obs_normalization
    )
    train_cfg.runner.pretrained_base_checkpoint = (
        str(checkpoint) if args.baseline else ""
    )
    isaac_args = make_isaac_args(args)

    env, env_cfg = task_registry.make_env(
        name="go2_arm", args=isaac_args, env_cfg=env_cfg
    )
    if args.terrain == "rough":
        prepare_rough_terrain(env)
    else:
        prepare_flat_terrain(env)
    env.cfg.terrain.curriculum = False

    runner, _ = task_registry.make_alg_runner(
        env=env,
        args=isaac_args,
        train_cfg=train_cfg,
        log_root=None,
    )
    if not args.baseline:
        runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    if hasattr(runner.obs_normalizer, "eval"):
        runner.obs_normalizer.eval()

    env.ppo_it = max(
        1000,
        int(
            checkpoint.stem.split("_")[-1]
            if args.baseline
            else runner.current_learning_iteration
        ),
    )
    env.keep_arm_fixed = args.fixed_arm
    env.arm_trajectory_curriculum.enabled = False
    env.cfg.terrain.curriculum = False

    set_seed(args.seed)
    observations, _ = env.reset()
    env.episode_length_buf[:] = torch.randint(
        0,
        int(env.max_episode_length),
        (env.num_envs,),
        device=env.device,
    )
    targets = torch.zeros(env.num_envs, 2, device=env.device)
    speeds = torch.full((env.num_envs,), 0.7, device=env.device)
    sample_base_targets(
        env, torch.arange(env.num_envs, device=env.device), targets, speeds
    )
    set_base_commands(env, targets, speeds)
    env.compute_observations()
    observations = env.obs_buf

    total = {
        "steps": 0,
        "episodes": 0,
        "fall": 0,
        "timeout": 0,
        "linear_error_sq": 0.0,
        "angular_error_abs": 0.0,
        "speed_error_abs": 0.0,
        "mechanical_energy": 0.0,
        "distance": 0.0,
        "arm_position_error": 0.0,
        "arm_orientation_error": 0.0,
        "base_height": 0.0,
        "targets_reached": 0,
        "reward": 0.0,
        "front_stance_width": 0.0,
        "rear_stance_width": 0.0,
        "left_wheelbase": 0.0,
        "right_wheelbase": 0.0,
        "minimum_foot_distance": 0.0,
        "low_base_steps": 0,
        "healthy_steps": 0,
        "healthy_base_height": 0.0,
        "arm_trajectory_finished_steps": 0,
        "arm_position_in_tol_steps": 0,
        "arm_orientation_in_tol_steps": 0,
        "arm_hold_steps": 0,
        "arm_settle_events": 0,
        "moving_arm_completion_events": 0,
        "static_arm_completion_events": 0,
        "joint_pass_events": 0,
        "arm_trajectory_attempts": int(env.num_envs),
    }

    for _ in range(args.steps):
        set_base_commands(env, targets, speeds)
        normalized_obs = runner.obs_normalizer(observations)
        actions = policy(normalized_obs)
        previous_xy = env.root_states[:, :2].clone()
        previous_target_quat = env.arm_traj_end_quat.clone()
        previous_target_lpy = env.arm_traj_end_lpy.clone()
        (
            observations,
            _,
            rewards,
            dones,
            _,
            _,
            _,
        ) = env.step(actions)

        valid = ~dones.bool()
        target_changed = (
            (
                1.0
                - torch.abs(
                    (
                        previous_target_quat
                        * env.arm_traj_end_quat
                    ).sum(dim=-1)
                )
            ).abs()
            > 1e-6
        ) | (
            torch.norm(
                previous_target_lpy - env.arm_traj_end_lpy,
                dim=-1,
            )
            > 1e-6
        )
        total["arm_trajectory_attempts"] += int(
            target_changed.sum().item()
        )
        count = int(valid.sum().item())
        if count > 0:
            linear_error = torch.norm(
                env.base_lin_vel[:, :2] - env.commands[:, :2],
                dim=-1,
            )
            angular_error = torch.abs(
                env.base_ang_vel[:, 2] - env.commands[:, 2]
            )
            speed_error = torch.abs(
                torch.norm(env.base_lin_vel[:, :2], dim=-1)
                - torch.norm(env.commands[:, :2], dim=-1)
            )
            mechanical_power = torch.relu(
                (env.torques * env.dof_vel).sum(dim=-1)
            )
            distance = torch.norm(
                env.root_states[:, :2] - previous_xy, dim=-1
            )
            total["steps"] += count
            total["linear_error_sq"] += (
                linear_error[valid].pow(2).sum().item()
            )
            total["angular_error_abs"] += angular_error[valid].sum().item()
            total["speed_error_abs"] += speed_error[valid].sum().item()
            total["mechanical_energy"] += (
                mechanical_power[valid].sum().item() * env.dt
            )
            total["distance"] += distance[valid].sum().item()
            total["arm_position_error"] += (
                env.arm_pos_err_norm[valid].sum().item()
            )
            total["arm_orientation_error"] += (
                env.arm_ori_err_norm[valid].sum().item()
            )
            total["base_height"] += (
                env.root_states[valid, 2].sum().item()
            )
            total["reward"] += rewards[valid].sum().item()
            total["arm_trajectory_finished_steps"] += int(
                (
                    env.arm_trajectory_finished[valid]
                ).sum().item()
            )
            total["arm_position_in_tol_steps"] += int(
                (env.arm_pos_in_tol[valid]).sum().item()
            )
            total["arm_orientation_in_tol_steps"] += int(
                (env.arm_ori_in_tol[valid]).sum().item()
            )
            total["arm_hold_steps"] += int(
                (env.arm_settle_in_tol[valid]).sum().item()
            )
            settle_event = (
                valid & env.moving_completion_event_buf
            )
            total["arm_settle_events"] += int(
                settle_event.sum().item()
            )
            moving_event = (
                settle_event
                & env.moving_completion_moving_event_buf
            )
            static_event = (
                settle_event
                & env.moving_completion_static_event_buf
            )
            total["moving_arm_completion_events"] += int(
                moving_event.sum().item()
            )
            total["static_arm_completion_events"] += int(
                static_event.sum().item()
            )
            joint_pass = (
                moving_event
                & (
                    env.moving_completion_event_mean_tracking
                    >= args.joint_min_tracking
                )
                & (
                    env.moving_completion_event_progress_ratio
                    >= args.joint_min_progress
                )
            )
            total["joint_pass_events"] += int(
                joint_pass.sum().item()
            )
            base_height = env._get_base_heights()
            healthy = (
                valid
                & (base_height > 0.18)
                & (torch.abs(env.roll) < 0.4)
                & (torch.abs(env.pitch) < 0.4)
            )
            healthy_count = int(healthy.sum().item())
            total["healthy_steps"] += healthy_count
            total["healthy_base_height"] += (
                base_height[healthy].sum().item()
            )
            total["low_base_steps"] += int(
                (base_height[valid] < 0.25).sum().item()
            )
            if healthy_count == 0:
                feet_xy = None
            else:
                feet_xy = env.feet_pos[:, :, :2]
                pair_distances = torch.stack(
                    [
                        torch.norm(feet_xy[:, 0] - feet_xy[:, 1], dim=-1),
                        torch.norm(feet_xy[:, 2] - feet_xy[:, 3], dim=-1),
                        torch.norm(feet_xy[:, 0] - feet_xy[:, 2], dim=-1),
                        torch.norm(feet_xy[:, 1] - feet_xy[:, 3], dim=-1),
                    ],
                    dim=-1,
                )
                total["front_stance_width"] += (
                    pair_distances[healthy, 0].sum().item()
                )
                total["rear_stance_width"] += (
                    pair_distances[healthy, 1].sum().item()
                )
                total["left_wheelbase"] += (
                    pair_distances[healthy, 2].sum().item()
                )
                total["right_wheelbase"] += (
                    pair_distances[healthy, 3].sum().item()
                )
                all_pairs = (
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 2),
                    (1, 3),
                    (2, 3),
                )
                minimum_distance = torch.stack(
                    [
                        torch.norm(
                            feet_xy[:, first] - feet_xy[:, second], dim=-1
                        )
                        for first, second in all_pairs
                    ],
                    dim=-1,
                ).min(dim=-1).values
                total["minimum_foot_distance"] += (
                    minimum_distance[healthy].sum().item()
                )

        done_ids = dones.nonzero(as_tuple=False).flatten()
        total["episodes"] += len(done_ids)
        if len(done_ids):
            total["fall"] += int(
                (
                    dones[done_ids].bool()
                    & ~env.time_out_buf[done_ids]
                ).sum().item()
            )
            total["timeout"] += int(
                env.time_out_buf[done_ids].sum().item()
            )

        distance_to_target = torch.norm(
            targets - env.root_states[:, :2], dim=-1
        )
        reached = (distance_to_target < 0.5) & valid
        reached_ids = reached.nonzero(as_tuple=False).flatten()
        total["targets_reached"] += len(reached_ids)
        reset_ids = done_ids
        resample_ids = torch.unique(
            torch.cat((reached_ids, reset_ids))
        )
        sample_base_targets(env, resample_ids, targets, speeds)

    steps = max(total["steps"], 1)
    healthy_steps = max(total["healthy_steps"], 1)
    episodes = max(total["episodes"], 1)
    distance = max(total["distance"], 1e-6)
    arm_attempts = max(total["arm_trajectory_attempts"], 1)
    result = {
        "mode": args.mode,
        "terrain": args.terrain,
        "checkpoint": str(checkpoint),
        "num_envs": env.num_envs,
        "steps_requested": args.steps,
        "valid_env_steps": total["steps"],
        "episodes": total["episodes"],
        "linear_velocity_rmse_mps": math.sqrt(
            total["linear_error_sq"] / steps
        ),
        "angular_velocity_mae_rad_s": (
            total["angular_error_abs"] / steps
        ),
        "speed_mae_mps": total["speed_error_abs"] / steps,
        "mean_mechanical_power_w": (
            total["mechanical_energy"] / (steps * env.dt)
        ),
        "mechanical_energy_j_per_m": (
            total["mechanical_energy"] / distance
        ),
        "arm_position_error_m": total["arm_position_error"] / steps,
        "arm_orientation_error_rad": (
            total["arm_orientation_error"] / steps
        ),
        "fall_rate": total["fall"] / episodes,
        "timeout_rate": total["timeout"] / episodes,
        "targets_reached_per_env": (
            total["targets_reached"] / env.num_envs
        ),
        "base_height_m": total["base_height"] / steps,
        "mean_reward_per_step": total["reward"] / steps,
        "front_stance_width_m": (
            total["front_stance_width"] / healthy_steps
        ),
        "rear_stance_width_m": (
            total["rear_stance_width"] / healthy_steps
        ),
        "left_wheelbase_m": total["left_wheelbase"] / healthy_steps,
        "right_wheelbase_m": total["right_wheelbase"] / healthy_steps,
        "minimum_foot_distance_m": (
            total["minimum_foot_distance"] / healthy_steps
        ),
        "healthy_step_fraction": total["healthy_steps"] / steps,
        "healthy_base_height_m": (
            total["healthy_base_height"] / healthy_steps
        ),
        "base_height_below_0_25_rate": (
            total["low_base_steps"] / steps
        ),
        "arm_trajectory_finished_fraction": (
            total["arm_trajectory_finished_steps"] / steps
        ),
        "arm_position_in_tol_fraction": (
            total["arm_position_in_tol_steps"] / steps
        ),
        "arm_orientation_in_tol_fraction": (
            total["arm_orientation_in_tol_steps"] / steps
        ),
        "arm_hold_fraction": total["arm_hold_steps"] / steps,
        "arm_trajectory_attempts": total["arm_trajectory_attempts"],
        "arm_settle_events": total["arm_settle_events"],
        "moving_arm_completion_events": (
            total["moving_arm_completion_events"]
        ),
        "static_arm_completion_events": (
            total["static_arm_completion_events"]
        ),
        "moving_completion_rate": (
            total["moving_arm_completion_events"] / arm_attempts
        ),
        "static_completion_rate": (
            total["static_arm_completion_events"] / arm_attempts
        ),
        "joint_pass_events": total["joint_pass_events"],
        "joint_pass_rate": total["joint_pass_events"] / arm_attempts,
        "joint_min_progress": args.joint_min_progress,
        "joint_min_tracking": args.joint_min_tracking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_path = args.output.parent / (
        f"{args.mode}_{args.terrain}.json"
    )
    result_path.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return result


def run_all(args):
    results = []
    for terrain in TERRAINS:
        for index, mode in enumerate(MODES):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                "--mode",
                mode,
                "--terrain",
                terrain,
                "--run-root",
                str(args.run_root),
                "--output",
                str(args.output),
                "--num-envs",
                str(args.num_envs),
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--device",
                "cuda:0",
                "--joint-min-progress",
                str(args.joint_min_progress),
                "--joint-min-tracking",
                str(args.joint_min_tracking),
            ]
            environment = os.environ.copy()
            environment["GO2_ARM_EXPERIMENT"] = mode
            environment["CUDA_VISIBLE_DEVICES"] = str(index % 2)
            print(f"\n=== Evaluating {mode} on {terrain} ===", flush=True)
            subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=True,
            )
            result_path = args.output.parent / f"{mode}_{terrain}.json"
            results.append(
                json.loads(result_path.read_text(encoding="utf-8"))
            )

    columns = (
        "linear_velocity_rmse_mps",
        "angular_velocity_mae_rad_s",
        "speed_mae_mps",
        "mean_mechanical_power_w",
        "mechanical_energy_j_per_m",
        "arm_position_error_m",
        "arm_orientation_error_rad",
        "fall_rate",
        "timeout_rate",
        "targets_reached_per_env",
        "base_height_m",
        "front_stance_width_m",
        "rear_stance_width_m",
        "left_wheelbase_m",
        "right_wheelbase_m",
        "minimum_foot_distance_m",
        "base_height_below_0_25_rate",
    )
    lines = [
        "| Method | Terrain | "
        + " | ".join(columns)
        + " |",
        "|---|---|" + "---:|" * len(columns),
    ]
    for result in results:
        lines.append(
            f"| {result['mode']} | {result['terrain']} | "
            + " | ".join(
                f"{result[column]:.5f}" for column in columns
            )
            + " |"
        )
    markdown = "\n".join(lines)
    args.output.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    print("\n" + markdown)


def main():
    args = parse_args()
    if args.all:
        run_all(args)
    elif args.single:
        evaluate_single(args)
    else:
        raise ValueError("Specify --all or --single")


if __name__ == "__main__":
    main()
