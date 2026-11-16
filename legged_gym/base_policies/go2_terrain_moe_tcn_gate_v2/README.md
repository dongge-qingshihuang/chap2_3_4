# Go2 Terrain-MoE TCN-Gate V2 Base Policy

This is the Chapter 3 pretrained base used by Chapter 4.

Source task:

```text
go2_terrain_moe_tcn_gate_v2
```

Source checkpoint:

```text
/home/hww/ws/chap3/legged_gym/logs/go2_terrain_moe_tcn_gate_v2/
Nov13_16-53-34_tcn_gate_v2_s0/model_14999.pt
```

Architecture:

```text
version:              go2_terrain_moe_v3_mlp_temporal_tcn_gate_norm
history:              20 x 45 = 900
experts:              4
latent dimension:     32
causal TCN layers:    3
gate output norm:     LayerNorm enabled
gate balance coef:    0.1
base action:          12
parameters:           2,604,968
```

The checkpoint contains no empirical observation normalization state.
