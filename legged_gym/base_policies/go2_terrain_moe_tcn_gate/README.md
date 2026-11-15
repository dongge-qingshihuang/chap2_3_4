# Go2 Terrain-MoE TCN-Gate Base Policy

This directory contains the frozen quadruped base policy migrated from the
local `chap3` project.

Source task:

```text
go2_terrain_moe_tcn_gate
```

Source checkpoint:

```text
/home/hww/ws/chap3/legged_gym/logs/go2_terrain_moe_tcn_gate/
Nov13_11-12-05_tcn_gate_s0/model_14999.pt
```

Migrated checkpoint:

```text
model_14999.pt
```

Architecture:

- 45 proprioceptive values over 20 history frames: `45 x 20 = 900`
- 4 predictive experts with EMA targets
- causal TCN temporal gate
- fused latent dimension 32
- 12-dimensional quadruped action output
- architecture version `go2_terrain_moe_v3_mlp_temporal_tcn_gate`

The `go2_arm` task keeps its independent six-frame, 486-dimensional arm
policy observation. The base policy receives its own 20-frame history, so
multi-critic, payload-awareness, IK-residual control, and arm rewards remain
unchanged.

The standalone base training task is registered as
`go2_terrain_moe_tcn_gate`.
