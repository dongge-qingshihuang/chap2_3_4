#!/usr/bin/env python3
"""Strictly validate the pretrained Terrain-MoE base checkpoint."""

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.modules import TerrainMoeActorCritic  # noqa: E402


PRETRAINED_MODEL_KWARGS = dict(
    num_modes=4,
    latent_dim=32,
    expert_hidden_dims=(256, 128, 64),
    actor_hidden_dims=(512, 256, 128),
    critic_hidden_dims=(512, 256, 128),
    gate_hidden_dims=(64, 64),
    dynamics_target_dim=30,
    contact_dim=4,
    future_contact_steps=15,
    terrain_dim=187,
    activation="elu",
    init_noise_std=0.8,
    ema_tau=0.01,
    gate_rho=0.3,
    gate_type="temporal_tcn",
    gate_tcn_channels=32,
    gate_tcn_layers=3,
    gate_temporal_dim=32,
    gate_kernel_size=3,
    gate_output_norm=True,
    gate_balance_coef=0.1,
    use_auxiliary_loss=True,
    train_perception_with_ppo=False,
    use_ema=True,
    history_encoder="mlp",
    history_encoder_hidden_dims=(64, 64, 64),
    tcn_width=192,
    tcn_layers=3,
    velocity_hidden_dims=(128, 64),
    adapter_hidden_dims=(64,),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "legged_gym"
            / "base_policies"
            / "go2_terrain_moe_tcn_gate_v2_obsnorm"
            / "model_14999.pt"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise TypeError(
            f"{checkpoint_path} is not a runner checkpoint with model_state_dict"
        )

    state_dict = checkpoint["model_state_dict"]
    metadata = checkpoint.get("architecture_metadata", {})
    model_kwargs = dict(PRETRAINED_MODEL_KWARGS)
    for key in (
        "num_modes",
        "latent_dim",
        "ema_tau",
        "gate_rho",
        "gate_type",
        "gate_tcn_channels",
        "gate_tcn_layers",
        "gate_temporal_dim",
        "gate_kernel_size",
        "gate_output_norm",
        "gate_balance_coef",
        "future_contact_steps",
        "history_encoder",
        "history_encoder_hidden_dims",
        "tcn_width",
        "tcn_layers",
    ):
        if key in metadata:
            model_kwargs[key] = metadata[key]
    history_len = int(metadata.get("history_len", 20))
    model = TerrainMoeActorCritic(
        45 * history_len,
        242,
        12,
        45,
        None,
        history_len,
        **model_kwargs,
    )
    destination = model.state_dict()
    compatible_state = dict(state_dict)
    if "std" in compatible_state and "log_std" in destination:
        compatible_state["log_std"] = compatible_state.pop("std").clamp(
            min=model.noise_std_min,
            max=model.noise_std_max,
        ).log()
    missing = sorted(set(destination) - set(compatible_state))
    unexpected = sorted(set(compatible_state) - set(destination))
    mismatched = sorted(
        key
        for key in set(destination) & set(compatible_state)
        if destination[key].shape != compatible_state[key].shape
    )
    load_result = model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        observation = torch.randn(2, 45 * history_len)
        actions = model.act_inference(observation)

    print(f"checkpoint: {checkpoint_path}")
    print(f"architecture_version: {checkpoint.get('architecture_version')}")
    print(f"actor_input_dim: {45 * history_len} (45 x {history_len})")
    print(f"actor_output_dim: {actions.shape[-1]}")
    print(f"parameter_count: {sum(p.numel() for p in model.parameters())}")
    print(f"missing_keys: {missing}")
    print(f"unexpected_keys: {unexpected}")
    print(f"shape_mismatches: {mismatched}")
    print(
        "action_noise_mapping: "
        f"std -> log_std (bounds "
        f"[{model.noise_std_min}, {model.noise_std_max}])"
    )
    print(f"strict_load: {load_result}")
    normalization_state = (
        checkpoint.get("obs_normalizer_state_dict")
        or checkpoint.get("obs_norm_state_dict")
    )
    print(f"normalization_state: {normalization_state is not None}")
    print(f"action_shape: {tuple(actions.shape)}")
    print(f"finite: {torch.isfinite(actions).all().item()}")


if __name__ == "__main__":
    main()
