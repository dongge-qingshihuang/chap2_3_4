# Chapter 4 Ablation Training Report

Snapshot time: 2026-11-20 Asia/Shanghai

> **Formal run status:** The first formal queue exposed an unbounded action
> standard deviation and was stopped. It remains the failure-analysis baseline.
> In the corrected bounded queue, B0 and B1 completed 8000 updates, while B4
> and B2 are running. The bounded Actor prevents the previous variance
> divergence.
>
> **Unified protocol:** every formal mode uses the calibrated load prior
> (`1.15/0.50` where applicable), a 1000-update fixed-arm warmup,
> `termination=-500`, and `stab__base_contact=-5`. Mode-specific ablations only
> disable the module named in the experiment table.

## Experiment Setup

- Robot: Go2 + ARX5 manipulator.
- Base policy: pretrained Terrain-MoE TCN-Gate v2 (`model_14999.pt`).
- PPO actor: 12-D Go2 action; the arm is trajectory-PD controlled.
- Environment count: 4096.
- Seed: 0.
- Reward scales are held constant across all ablation modes.
- Pre-calibration B1 log root:
  `legged_gym/logs/chapter4_ablation_queue_final_fixed`.
- Calibrated B0/B2/B3/B4/B0F log root:
  `legged_gym/logs/chapter4_ablation_queue_tuned`.
- Safety candidate log root:
  `legged_gym/logs/chapter4_ablation_queue_safety_warmup`.
- Formal run log root:
  `legged_gym/logs/chapter4_ablation_formal`.
- Bounded rerun log root:
  `legged_gym/logs/chapter4_ablation_formal_bounded`.

## Queue Status

| Mode | Description | Status |
|---|---|---|
| B0 | Full method (`Ours`) | Finished, 8000 updates |
| B1 | Single critic, equal fixed group weighting | Finished, 8000 updates |
| B4 | No pretrained initialization | Running, intermediate |
| B2 | No arm trajectory curriculum | Running, intermediate |
| B3 | No dynamic fusion; payload estimation retained | Pending in bounded rerun |
| B0F | Pretrained control with B4's full unfreeze scope | Pending in bounded rerun |

## P0 Correctness Fixes

The previous formal multi-critic curves were invalidated by three issues:

1. `TerrainMoeActorCritic` used a direct `std` parameter, while PPO only
   clamped an attribute named `log_std`.
2. B1 used a scalar sum that was not aligned with the fixed group weights.
3. B3 disabled the payload estimator together with dynamic fusion, creating a
   confounded ablation.

The corrected implementation now:

- Stores action noise as `log_std` and maps old checkpoint `std` keys into it.
- Enforces `0.05 <= std <= 0.5` after every PPO mini-batch update.
- Logs mean/min/max std, clamp fraction, actor entropy, and actor KL.
- Uses `0.7 task + 0.2 stability + 0.1 energy` for B1's scalar reward.
- Keeps payload estimation and supervision in B3 while disabling only dynamic
  fusion and falling back to fixed `[0.7, 0.2, 0.1]` advantage weights.

The V2 pretrained checkpoint remains compatible: strict loading reports no
missing, unexpected, or mismatched parameters after the `std -> log_std`
conversion.

The superseded 8000-update B0/B1 curves and intermediate B2/B4 results remain in
the failure-analysis sections below.

## Bounded Rerun: Final B0 vs B1

| Metric | B0 multi-critic | B1 single critic | Better |
|---|---:|---:|---|
| Mean episode length | 936.0 | 933.5 | B0, marginal |
| Hard-fall termination | 12.0% | 11.8% | Tie |
| Base-contact termination | 6.4% | 5.3% | B1, marginal |
| Timeout | 88.0% | 88.2% | Tie |
| Terrain level | 3.256 | 3.297 | B1, marginal |
| Linear tracking | 0.815 | 0.817 | Tie |
| Angular tracking | 0.409 | 0.415 | Tie |
| Arm position error | 0.0747 m | 0.0746 m | Tie |
| Arm orientation error | 0.3493 rad | 0.3519 rad | B0, marginal |
| Settle events/s | 0.0837 | 0.0682 | B0 |
| Curriculum success | 35.7% | 30.9% | B0 |
| Curriculum stability success | 56.7% | 53.4% | B0 |
| Task reward | 1.224 | 1.232 | Tie |
| Stability reward | -0.366 | -0.594 | B0 |
| Energy reward | -0.622 | -1.460 | B0 |
| Expert-gate entropy | 0.796 | 0.785 | Tie |
| Mean action std | 0.381 | 0.377 | Tie |
| Actor KL, last-100 mean | 0.015 | 11.50 | B0 |

The completed methods are close on locomotion and terrain, with a trade-off:

- B0 has higher curriculum success/stability, more settle events, and much
  lower stability/energy penalties.
- B1 is only marginally better on terrain and velocity tracking; arm errors are
  effectively tied.
- The scalar `Train/mean_reward` values (`8.98` for B0 and `14.44` for B1) are
  not directly comparable because B0 fuses normalized critic advantages while
  B1 optimizes a fixed weighted raw group sum.

Both bounded policies finish with action standard deviation near `0.38` and no
active clipping, so the variance divergence is fixed. B1's late Actor KL is
nevertheless very large (`11.5` over the last 100 points), while B0 remains near
`0.015`; B1's apparent performance advantage therefore needs deterministic
evaluation and KL-stability review.

## Deterministic Waypoint Evaluation

Protocol:

- Final `model_8000.pt` for every mode.
- 64 environments, 1200 policy steps, seed `20261118`.
- Random base waypoints in each terrain platform and random IK arm targets.
- Domain randomization, observation noise, pushes, and delayed actions disabled.
- Flat uses terrain column 0. Rough uses columns
  `2,4,6,7,8,10,12,14,16,18,19` at row 3, covering rough flat, down-slope,
  rough up/down slopes, stairs up/down, and discrete obstacles. The smooth
  up-slope column and terrain rows other than row 3 were not included in this
  first evaluation.
- Mechanical energy is positive joint work; energy efficiency is Joules per
  travelled metre.

| Method | Flat speed RMSE | Flat energy J/m | Flat fall | Rough speed RMSE | Rough energy J/m | Rough fall | Rough arm pos | Rough targets/env |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.274 | **71.00** | **1.19%** | 0.334 | 95.26 | 17.39% | 0.159 m | 2.17 |
| B1 | 0.279 | 79.29 | 3.57% | 0.331 | 96.22 | 20.00% | **0.150 m** | 2.05 |
| B2 | **0.235** | 81.07 | 1.20% | **0.320** | 101.81 | 5.81% | 0.182 m | **2.16** |
| B3 | 0.282 | 74.98 | **0.00%** | 0.333 | 97.06 | 11.24% | 0.170 m | 1.91 |
| B4 | 0.296 | 71.16 | 2.35% | 0.340 | **91.86** | 14.44% | 0.169 m | 1.89 |
| B0F | 0.278 | 79.16 | 2.38% | 0.322 | 92.13 | **3.53%** | 0.155 m | 2.05 |

Conclusions from this evaluation:

- On flat terrain, B0 is the best overall compromise: lowest mechanical energy,
  lowest fall rate tied with B2, and competitive velocity/arm errors.
- On rough terrain, B0F is the safest and most balanced policy: lowest fall
  rate, second-lowest energy, and low arm-position error.
- B2 has the best flat velocity tracking and strong rough velocity tracking,
  but its rough energy efficiency and arm accuracy are the worst.
- B4 has slightly lower rough energy than B0F, but much higher fall rate and a
  lower base height, so its apparent energy advantage is partly caused by a
  lower/less mobile gait.
- Full B0 is not robust on rough terrain: its fall rate is `17.39%`, far worse
  than B0F's `3.53%`. This supports retaining pretrained initialization.

Machine-readable results:
`legged_gym/logs/chapter4_evaluation/results.json` and
`legged_gym/logs/chapter4_evaluation/results.md`.

## New Baseline 111: Pose Audit and Training Configuration

The next training baseline is:

`legged_gym/logs/111/model_14999.pt`

Checkpoint properties:

- `iter=14999`
- Terrain-MoE TCN-Gate normalized architecture
- actor, critic, gate, adapters, online experts, and velocity estimator present
- legacy `std` key successfully converted to bounded `log_std`
- strict load: no missing, unexpected, or mismatched parameters
- no stored observation-normalization state

Pose audit under the same random-waypoint protocol:

| Terrain | Base height | Base below 0.25 m | Front stance width | Rear stance width | Left wheelbase | Right wheelbase | Fall rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 0.206 m | 92.9% | 0.182 m | 0.187 m | 0.379 m | 0.356 m | 44.4% |
| Rough | 0.061 m | 87.0% | 0.190 m | 0.194 m | 0.384 m | 0.357 m | 81.2% |

The base-height problem is clear: baseline 111 is far below the configured
`0.32 m` target, including on flat terrain. The rough-terrain result is even
worse because most episodes have already fallen.

The stance widths are not narrower than the completed B0 policy: final B0 had
front/rear widths of `0.171/0.148 m`, whereas baseline 111 has
`0.182/0.187 m`. Therefore the audit does not support the claim that 111 has an
unusually narrow stance relative to the existing policies, although a separate
nominal URDF stance benchmark should be added if a hard engineering threshold
is required.

Training configuration changes:

- All pretrained modes now load baseline 111 by default.
- `CHAPTER4_BASE_CHECKPOINT` can override the checkpoint path.
- `empirical_normalization=True`; the observation normalizer starts from
  identity because checkpoint 111 contains no normalization statistics, then
  learns online and is saved in subsequent checkpoints.

The Chapter 4 pipeline with the new checkpoint and normalization path passes.

### Fixed-arm base-only audit

> **Superseded:** this run was performed after empirical normalization was
> enabled globally but before the evaluator froze the normalizer. Because
> checkpoint 111 contains no normalization statistics, the normalizer updated
> during evaluation and changed the policy input distribution. The numbers in
> this subsection are invalid and are not used for conclusions.

The arm was held at its default PD target while only the Go2 base was driven by
random waypoints. The physical arm and its mass remained attached.

| Terrain | Mean base height | Healthy base height | Healthy steps | Fall | Front width | Rear width | Min foot distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 0.217 m | 0.287 m | 43.8% | 99.4% | 0.308 m | 0.294 m | 0.196 m |
| Rough | 0.082 m | 0.294 m | 44.6% | 99.5% | 0.299 m | 0.294 m | 0.194 m |

Conclusions:

- Foot spacing is not too narrow. In healthy upright steps the front/rear
  stance widths are approximately `0.30 m`, with a minimum inter-foot distance
  near `0.20 m`.
- Base height is moderately low even when upright (`0.287-0.294 m` versus the
  `0.32 m` target), but the raw means (`0.082-0.217 m`) are dominated by falls.
- The dominant problem is that baseline 111 is not stable when the arm is
  frozen: fall rate is about `99.5%` on both flat and rough terrain.
- This conflicts with the existing 1000-update fixed-arm warmup. If baseline
  111 must be used, the fixed-arm warmup should be shortened, removed, or the
  baseline should first be adapted to a static arm before full Chapter 4
  training.

### Full B0 pose audit

Checkpoint:
`chapter4_ablation_formal_bounded/.../B0_v2_s0/model_8000.pt`.

Observation normalization was disabled for this evaluation because the
checkpoint predates empirical normalization and has no stored statistics.
Metrics below use healthy upright steps for stance geometry.

| Terrain | Mean base height | Healthy base height | Healthy steps | Base below 0.25 m | Fall | Front width | Rear width | Left wheelbase | Right wheelbase | Min foot distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat | 0.308 m | 0.308 m | 98.9% | 0.07% | 1.2% | 0.169 m | 0.147 m | 0.335 m | 0.294 m | 0.123 m |
| Rough | 0.171 m | 0.314 m | 98.1% | 1.72% | 17.2% | 0.182 m | 0.160 m | 0.334 m | 0.304 m | 0.130 m |

Conclusions:

- Base height is **not low while the policy remains upright**:
  `0.308 m` flat and `0.314 m` rough, close to the `0.32 m` target.
- The low rough-terrain mean base height is caused mainly by the `17.2%` fall
  rate, not by a systematically collapsed standing posture.
- Foot spacing is narrow relative to baseline 111. The final B0 front/rear
  stance widths are only `0.169/0.147 m` flat and `0.182/0.160 m` rough.
- The minimum distance between any two feet is only `0.123-0.130 m`, indicating
  a compact support polygon and increased fall risk under arm load.

Raw audit files:
`legged_gym/logs/chapter4_evaluation_full_pose/B0_flat.json` and
`legged_gym/logs/chapter4_evaluation_full_pose/B0_rough.json`.

### Full training with baseline 111 and foot-spacing reward

The `_get_base_heights()` audit confirms that the reported height is the Go2
base height:

```text
root_states[:, 2] - terrain_height_under_base_footprint
```

`root_states` belongs to the root/base rigid body. The ARX5 end-effector,
arm links, and arm base mount are not part of this calculation.

New stability reward:

- `stab__foot_spacing = -2.0`
- target front/rear lateral stance width: `0.24 m`
- target minimum distance between any two feet: `0.18 m`
- positive normalized squared shortfalls are multiplied by the negative scale,
  so narrow stances receive a penalty

The first implementation accidentally returned a negative shortfall and was
stopped after two updates. A corrected run was launched at
`legged_gym/logs/chapter4_full_111_foot_spacing_v2`, but it was superseded when
the baseline was corrected to the Chap3 obs-normalized LiftSpacing checkpoint.
Its log artifacts were removed in the 2026-11-20 cleanup. The active replacement
run is:

`legged_gym/logs/chapter4_full_obsnorm_warmup_v6`

Training configuration for this full run:

- baseline:
  `legged_gym/base_policies/go2_terrain_moe_tcn_gate_v2_obsnorm/model_14999.pt`
- empirical observation normalization: enabled
- mode: B0 full method
- updates: 8000
- corrected foot-spacing reward is present in logs as
  `Episode/rew_stab__foot_spacing`

### Reward and environment audit

Checks that pass:

- Every non-zero configured reward scale resolves to an existing reward
  function.
- Task, stability, energy, base-contact, termination, and foot-spacing terms are
  assigned to the intended critic groups.
- Timeout is excluded from the hard-fall termination reward.
- `_get_base_heights()` uses the Go2 root/base rigid-body height, not the arm.
- Foot-spacing reward sign is now correct: positive shortfall times negative
  scale gives a penalty.
- At a synthetic stance with `0.24 m` lateral width and at least `0.18 m`
  inter-foot distance, the foot-spacing penalty is exactly zero.

Remaining issues or caveats:

1. `_reward_arm_action_rate()` and `_reward_arm_smoothness()` currently operate
   on `self.actions`, which contains the 12-D Go2 actions, not arm targets.
   These terms are not enabled in the current reward scales, but they should not
   be activated without correction.
2. `_reward_joint_power()` penalizes squared signed mechanical power. This is a
   smoothing/shaping cost, not literal energy consumption in Joules. This is
   consistent with the original design but should be named accordingly in paper
   descriptions.
3. Foot-spacing is evaluated every control step, including swing. It therefore
   shapes both support width and swing-foot separation rather than only the
   support polygon.

### Obs-normalized checkpoint migration audit

The file originally placed in `logs/111` was verified by SHA-256 to be the
Chap3 **non-obsnorm** `foot_lift_spacing` checkpoint:

```text
88cca34ddc4967d914211e908b64c60d7a26c04ac036f0563d76c97333a32758
```

The intended obs-normalized baseline is:

```text
chap3/logs/go2_terrain_moe_tcn_gate_v2_obsnorm/
Nov19_12-19-47_obsnorm_lift_spacing_s0/model_14999.pt
```

It was copied into:

`legged_gym/base_policies/go2_terrain_moe_tcn_gate_v2_obsnorm/model_14999.pt`

Migration checks:

- actor/perception transfer: 271 parameters, strict verification passed.
- old `std` converted to bounded `log_std`.
- 45-D observation-normalization state found and loaded.
- normalizer sample count: `100024320`.
- source command semantics restored to `heading_command=True`.
- Chapter 4 pipeline completes successfully.

The checkpoint itself is healthy in its source environment. With the confirmed
`heading_command=True` training semantics, Chap3 row-3 stairs-up, stairs-down,
slope-up, and slope-down evaluation achieved 100% success and 0% fall.

However, behavioral transfer to the Go2+ARX5 environment is not immediate:

- within 5 PPO updates, fall termination reached about 95%;
- roll termination remained above 80%;
- base height fell to roughly 0.19 m;
- linear tracking fell to roughly 0.10-0.15.

Therefore:

```text
File migration:              successful
Weight migration:            successful
Normalization migration:     successful
Behavioral morphology transfer: failed
```

The immediate behavioral failure is consistent with the checkpoint being
trained on the Go2-only URDF while the Chapter 4 environment adds the ARX5
mass, inertia, and changing arm pose. Direct adaptation is now evaluated through
the existing 1000-update fixed-arm warmup together with the zero-initialized
arm-reference residual described below; these startup metrics are therefore not
by themselves evidence that the checkpoint cannot be adapted.

### B3R arm-reference base-action residual

`B3R` keeps the Chap3 actor deterministic output as the base leg action and
adds a small learned correction:

```text
leg action = Chap3 actor mean + bounded residual
```

The residual branch uses:

- the 80-D B3 actor feature: current 45-D proprio observation, 32-D fused
  control latent, and 3-D velocity estimate;
- the current 6-D arm joint position error;
- the current 6-D arm joint velocity;
- the current control step's 6-D IK-reference joint velocity already computed
  by the environment as `arm_target_vel`.

The 18-D arm inputs use the environment's existing position and velocity
observation scales and are stored per transition for the PPO update. The final
residual layer is zero-initialized. Therefore the initial deterministic policy
is exactly the Chap3 actor output. Residuals are bounded by
`0.1 * tanh(...)` in action units, corresponding to at most `0.025 rad` of leg
joint-target correction at the current `action_scale=0.25`.

The 9-D estimated payload is deliberately excluded from this first version and
remains a separate future increment. The original 900-D actor history layout is
unchanged; only the side-channel 18-D arm state is stored alongside the action.

The first version uses B3's non-dynamic critic configuration with fixed
`[0.7, 0.2, 0.1]` task/stability/energy advantage weights. `B3R` keeps the
pretrained Chap3 actor, control adapters, gate, temporal encoder, and action
standard deviation frozen. `B3RM` instead fine-tunes the pretrained actor,
control adapters, gate, action standard deviation, online experts, and velocity
estimator jointly with the residual. The EMA target experts remain frozen. The
payload estimator continues to train through its dedicated supervision loss.

The four controlled groups are:

- `B3`: original fixed-weight reference;
- `B3R`: B3 plus the arm-conditioned residual;
- `B3M`: B3 plus the moving-arm completion reward;
- `B3RM`: B3 plus both modules and full pretrained-base fine-tuning.

Smoke checks pass for zero-output initialization, bounded residual output,
residual-only gradients, storage round-trip, checkpoint initialization, joint
completion reward edge cases, and a full environment/PPO update.

The earlier run at `chapter4_b3r_armref_residual_v2` used the superseded
`9-D payload + 18-D arm + 0.5 clip` implementation. It was stopped and is not a
result for the current method. No replacement training run has been launched.

### Moving-arm completion reward

`task__moving_arm_completion` fires once per arm trajectory after:

- the local reference trajectory has finished;
- position and orientation remain inside their current settle tolerances for
  15 consecutive control steps;
- the step is not a reset or termination;
- the trajectory has not already received this event.

For the same 15-step window, the implementation accumulates:

- mean linear velocity tracking reward;
- signed displacement along the active command direction;
- commanded distance.

The raw event reward is:

```text
mean_linear_tracking * clip(signed_progress / required_progress, 0, 1)
```

The executed command is snapshotted at the beginning of each physics step before
command resampling. Tolerance interruption, target replacement, and reset clear
the window. Near-zero commanded distance receives zero reward and is logged as a
separate static completion. With `b=0.2`, the configured scale is `b/dt=10`, so
after the standard reward-scale `dt` multiplication one full event contributes
exactly `0.2`.

The independent evaluator now also records trajectory-finished, position,
orientation, hold, settle-event, moving/static completion, and joint-pass
metrics. Joint pass requires the moving completion event, mean tracking at least
`0.7`, and progress ratio at least `0.8` by default.

Dashboard scalars:

- `BaseActionResidual/mean_abs`
- `BaseActionResidual/max_abs`
- `BaseActionResidual/saturation_fraction`
- `Episode/moving_arm_completion_events`
- `Episode/static_arm_completion_events`
- `Episode/moving_arm_completion_bonus_per_event`

`max_abs` is the maximum over all environments and leg joints and therefore
approaches the bound as soon as one value is strongly saturated. It should not
be interpreted as the whole residual being clipped. The useful diagnostic is
`saturation_fraction`, the fraction of residual elements inside one percent of
the bound.

## Superseded Formal Snapshot

| Metric | B0 final | B1 final | B4 at ~4860 | B2 at ~5860 |
|---|---:|---:|---:|---:|
| Mean reward | -15.00 | **+15.73** | -89.32 | -16.03 |
| Episode length | 949.22 | **958.74** | 814.20 | 928.90 |
| Hard-fall termination | **6.1%** | 9.3% | 28.6% | 8.5% |
| Base-contact termination | 0.0% | 4.6% | 8.9% | 3.4% |
| Terrain level | 0.00 | **2.80** | 0.00 | 0.00 |
| Linear tracking | 0.181 | **0.743** | 0.174 | 0.183 |
| Angular tracking | 0.140 | **0.419** | 0.183 | 0.146 |
| Arm position error | 0.220 m | **0.073 m** | 0.070 m | 0.353 m |
| Arm orientation error | 0.758 rad | **0.351 rad** | 0.274 rad | 1.034 rad |
| Curriculum success | 8.8% | **28.0%** | 11.9% | N/A |
| Curriculum stability success | 12.2% | **50.3%** | 4.5% | N/A |
| Action noise std | **10.99** | 0.26 | 10.70 | 8.47 |

B0 had a valid warmup phase, but its policy standard deviation expanded after
the fixed-arm warmup ended. At iteration 3000, mean reward reached roughly
-683 and energy reward roughly -38. The policy partially recovered numerically
by iteration 8000, but terrain level remained zero and velocity tracking stayed
near 0.18/0.14. This is not a successful full-method result.

B4 is still an intermediate from-scratch run and already shows poor terrain and
expert routing. B2 remains weak in arm tracking without the trajectory
curriculum. These runs are not yet final.

### Curve-phase diagnosis

The formal curves separate into three regimes.

**0-1000: fixed-arm warmup.** B0 and B1 both keep long episodes and positive
task signal while the arm remains at its default pose. This validates the
warmup itself.

**1000-3000: post-warmup instability.** B1 incurs a short transition cost and
then recovers. B0 instead degrades continuously:

| Iteration | B0 mean reward | B0 action std | B0 terrain | B1 mean reward | B1 action std | B1 terrain |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | -87.5 | 1.45 | 0.00 | 5.21 | 0.55 | 0.00 |
| 1500 | -255.5 | 2.50 | 1.30 | 1.88 | 0.58 | 1.28 |
| 2000 | -463.3 | 3.78 | 2.12 | 0.37 | 0.61 | 2.35 |
| 2500 | -694.5 | 5.38 | 1.03 | 2.18 | 0.59 | 2.41 |
| 3000 | -683.2 | 6.38 | 0.00 | 4.26 | 0.54 | 2.45 |

**3000-8000: numerical recovery without task recovery.** B0's loss and mean
reward stop diverging, but the action standard deviation continues to rise to
`10.99`, expert entropy repeatedly collapses, terrain returns to zero, and
linear/angular tracking finish near `0.18/0.14`. The final policy has long
episodes mainly because it times out at a degenerate low-speed/low-height
behavior rather than because it locomotes successfully.

The same variance pathology appears in the intermediate multi-critic runs:

- B2 reaches action std `8.67` by iteration 5977.
- B4 reaches action std `10.84` by iteration 4959.
- B1 remains near `0.26`, with expert entropy near `0.77` and terrain level
  `2.80`.

This pattern points to a shared multi-critic optimization problem rather than
only a B0 load-weight problem. `TerrainMoeActorCritic` stores action noise in
`std`, while the optimizer currently attempts to clamp only an attribute named
`log_std`; the `std` parameter therefore has no effective upper bound.
Post-warmup distribution shift then provides the perturbation that lets the
unbounded multi-critic policy variance diverge.

The immediate engineering priority is:

1. Clamp the actual `actor_critic.std` parameter after every optimizer step.
2. Log the clamp fraction and actor entropy.
3. Smooth or stage the transition out of fixed-arm warmup.
4. Treat current multi-critic B0/B2/B4 curves as diagnostics rather than final
   paper results.

## Shared Load-Prior Calibration

The previous B0 fusion weight was approximately
`[Task 0.255, Stability 0.681, Energy 0.064]`. Its load risk was already
`1.13-1.73`, so the original threshold `0.5` left the stability prior
saturated and the actual stability weight nearly constant.

All load-aware modes now use:

- `LOAD_RISK_THRESHOLD = 1.15`
- `LOAD_STAB_WEIGHT_MAX = 0.50`

Additional diagnostics are logged as:

- `Critic/{task,stability,energy}_raw_advantage_std`: standard deviation before
  per-critic advantage normalization.
- `Fusion/{task,stability,energy}_component_rms`: RMS of each weighted,
  normalized critic contribution.
- `Fusion/fused_advantage_std`: standard deviation of the final fused advantage.

## Unified Safety Protocol

All formal modes use:

- `termination = -500.0`: a physical-failure terminal penalty; rewards are
  multiplied by `dt=0.02`, so the terminal step contributes `-10`.
- `stab__base_contact = -5.0`: contributes `-0.1` for each step with a base or
  other termination-body contact.

Termination causes are now logged separately:

- `Episode/roll_termination_rate`
- `Episode/pitch_termination_rate`
- `Episode/base_contact_termination_rate`
- `Episode/fall_termination_rate`
- `Episode/timeout_rate`

At the start of the preliminary B0S run, most short episodes were caused by
roll termination rather than base contact. The agent subsequently learned out
of that initial instability.

The preliminary candidate was stopped at 2648 updates after validating the
safety settings, which were then promoted to the formal protocol.

### Preliminary B0S result

At approximately iteration 2600:

| Metric | B0S |
|---|---:|
| Mean reward | -9.69 |
| Episode length | 933.31 |
| Roll termination | 4.5% |
| Pitch termination | 1.1% |
| Base-contact termination | 7.9% |
| Any hard-fall termination | 13.5% |
| Timeout | 86.5% |
| Terrain level | 3.00 |
| Task reward | 1.16 |
| Stability reward | -0.50 |
| Energy reward | -1.47 |
| Linear tracking | 0.78 |
| Angular tracking | 0.37 |
| Curriculum success | 32.6% |
| Curriculum stability success | 54.4% |
| IK success | 80.3% |

The safety warnings and fixed-arm warmup materially improved episode length:
the no-safety calibrated B0 had roughly 430-630 steps and 55-80% base-contact
termination, while B0S has roughly 900-950 steps and less than 10% base-contact
termination near iteration 2600.

The trajectory curriculum is still progressing slowly. At the snapshot, about
70% of newly sampled trajectories were in `local`, 30% in `medium`, and almost
none had reached `large` or `full`, because the 60% success-rate promotion
threshold was not consistently met.

B1 finished all 8000 updates on the older pre-warmup protocol with mean reward
`16.81`, episode length `861.74`, terrain level `2.69`, base-contact termination
`5.0%`, linear tracking `0.796`, and angular tracking `0.422`.

## Arm Trajectory Curriculum

There are four trajectory stages:

1. `local`: minimum iteration 0.
2. `medium`: minimum iteration 200.
3. `large`: minimum iteration 500.
4. `full`: minimum iteration 1000.

Promotion additionally requires at least 20 attempts and both success and
stability rates of at least 60%, so the minimum iterations are eligibility
times rather than guaranteed transition times.

Before these stages begin, the arm now has an independent 1000-update fixed
warmup. During warmup:

- The command is generated from the default arm joint configuration.
- The PD target remains exactly at the default arm pose.
- Target resampling, trajectory advancement, settle events, and curriculum
  promotion are disabled.
- At iteration 1000, all environments receive their first local-curriculum
  target.

`Episode/arm_warmup_active` records whether the fixed arm warmup is active.

## Metric Definitions

- **Mean reward**: mean cumulative scalar episode return.
- **Episode length**: mean number of policy steps; configured maximum is 1000.
- **Task/Stability/Energy**: normalized mean group reward from
  `Episode/rewgroup_*`.
- **Settle success rate**: successful arm settle events divided by trajectory
  attempts. This is not the success rate of the whole locomotion task.
- **Stability success rate**: fraction of arm trajectory attempts that also
  satisfy the base stability condition.
- **IK success**: fraction of sampled arm targets accepted by the reachability
  and body-collision filters.
- **Terrain level**: mean terrain curriculum level at environment reset.

## Historical Provisional Metrics

| Mode | Mean reward | Episode length | Return/step | Task | Stability | Energy | Arm pos error | Arm ori error | Settle/s | Settle success | Stability success | IK success | Terrain level |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1.30 | 930.74 | 0.00140 | 0.290 | -0.0357 | -0.399 | 0.0638 m | 0.3535 rad | 0.0345 | 32.95% | 52.99% | 77.24% | 0.00 |
| B1 | **18.50** | 917.01 | **0.02018** | — | — | — | 0.0765 m | 0.3704 rad | 0.0379 | 20.38% | 44.30% | 80.63% | 3.13 |
| B2 | -4.93 | 708.93 | -0.00695 | 0.286 | -0.0740 | -0.946 | 0.1981 m | 0.5688 rad | 0.0627 | — | — | — | 0.00 |
| B3 | 6.93 | **979.09** | 0.00708 | **1.236** | -0.0807 | -0.790 | 0.0736 m | **0.3531 rad** | **0.0525** | 21.71% | 46.27% | **83.09%** | **3.40** |
| B4 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Group reward columns are unavailable for B1 because the three groups are
intentionally merged into one critic.

## Dynamic Fusion and Expert Gate

| Mode | Task weight | Stability weight | Energy weight | Load-gate entropy | Expert-gate entropy |
|---|---:|---:|---:|---:|---:|
| B0 | 0.255 | 0.681 | 0.064 | 0.786 | 0.668 |
| B1 | — | — | — | — | 0.818 |
| B2 | ~0.244 | ~0.695 | ~0.061 | 0.767 | ~0.02 |
| B3 | fixed 0.700 | fixed 0.200 | fixed 0.100 | — | 0.841 |
| B4 | Pending | Pending | Pending | Pending | Pending |

B3 uses fixed weights by design. Its current running process predates a
diagnostic-only logging fix, so its saved `LoadGate/actual_*` scalars may appear
as zero; the PPO fusion itself uses the fixed `[0.7, 0.2, 0.1]` weights.

## Historical Provisional Findings

The following findings describe interrupted pre-formal queues. They are kept
for diagnosis and must not be used as the final comparison.

### B1: highest scalar return

B1 has the highest mean reward, `18.50`, and the highest return per step,
`0.02018`. Its component logs show that this comes mainly from stronger Go2
velocity tracking and lower energy costs:

```text
linear velocity tracking:  B1 0.819  vs B0 0.165
angular velocity tracking: B1 0.438  vs B0 0.125
joint power cost:          B1 -0.088 vs B0 -0.176
```

This is a valid scalar-return comparison because all modes use the same reward
formula and coefficients. However, B1 does not expose separate critic-level
Task/Stability/Energy returns.

### B3: best current multi-critic result

Among multi-critic modes, B3 currently has the highest return, longest
episodes, and highest task reward. It also reaches terrain level 3.40. Its
fixed task-heavy weights lead to stronger locomotion tracking than B0.

### B0: best stability and precision trade-off

B0 has the lowest arm position error, `0.0638 m`, the highest stability success
rate, `52.99%`, and the lowest energy penalty. Its learned fusion weights are
stability-heavy at roughly `[0.255, 0.681, 0.064]`.

This produces a conservative policy: stable and energy-efficient, but with
weaker velocity tracking and no terrain-curriculum progress.

### B0 terrain level remains zero

The flat warmup ends after 1000 updates, so this is not a permanent warmup
lock. The terrain curriculum requires the robot to move more than half a
terrain tile, about 4 m, before promotion. Otherwise the `move_down` condition
can reduce the terrain level back to zero.

The B0 policy has much weaker velocity tracking than B1/B3 and therefore does
not generate enough displacement. This is behaviorally caused by the
stability-heavy learned fusion, not by a terrain configuration switch.

### B2: curriculum is important

Without the arm trajectory curriculum, B2 has the shortest episodes and the
largest arm tracking errors:

```text
B2 arm position error: 0.1981 m
B2 arm orientation error: 0.5688 rad
```

Its Expert Gate entropy also collapses close to zero in the observed window,
indicating concentration on very few experts. The ordered curriculum appears
important for both arm tracking and stable expert routing.

## Success-Rate Interpretation

B1 has a lower Settle success rate than B0, despite a much higher scalar
return. This is not contradictory:

- B1's return is dominated by velocity tracking and energy savings.
- Arm Settle success is not a Task Critic reward.
- B1 prioritizes faster locomotion and lower energy, which can make the arm
  endpoint harder to hold inside the settle tolerances.
- `Settle/s` and `Settle success` have different denominators: event rate over
  simulation time versus successful events over trajectory attempts.

Therefore, B1 is not uniformly better. It is better on the scalar locomotion
return, while B0 is better on arm precision, stability success, and terrain
motion conservatism.

## Current Ranking

```text
Mean reward:          B1 > B3 > B0 > B2
Episode length:       B3 > B0 > B1 > B2
Settle success:       B0 > B3 > B1
Stability success:    B0 > B3 > B1
Arm position error:   B0 < B3 < B1 < B2
Terrain progress:     B3 > B1 > B0 = B2
```

The current experimental evidence does not yet show the full B0 method
outperforming fixed-weight B3. It shows a clear trade-off: dynamic load-aware
fusion improves stability and energy discipline but currently suppresses the
locomotion task objective too strongly.

## Validity and Caveats

- The pre-calibration B0 and calibrated non-safety B0 were interrupted and
  retained only as diagnostic baselines.
- The preliminary B0S run stopped at 2648 updates after validating the common
  safety and warmup protocol.
- The formal `B0,B1,B4,B2,B3,B0F` queue is running from zero for 8000 updates.
- All six formal modes use the same safety rewards and fixed-arm warmup.
- B0 and B1 are complete. B0 is invalid as a successful full-method result
  because its policy standard deviation and return diverged after warmup.
- B4 and B2 are intermediate; B3 and B0F have not started.
- The reported values are training metrics, not deterministic evaluation.
- The final comparison must use the same completed iteration window.
- B1's success rate is a curriculum metric and should not be compared directly
  with `Train/mean_reward`.
- No final conclusion about pretrained initialization is possible before B4.

## Recommended Follow-Up

1. Let the formal `B0,B1,B4,B2,B3,B0F` queue finish without changing the shared
   reward or warmup protocol.
2. Run identical deterministic evaluations for all modes, including fixed
   command tracking, fall rate, arm pose error, settle rate, and energy.
3. For a B0 follow-up, raise the Task weight floor or lower the Stability prior
   maximum to reduce the current stability-heavy behavior.
4. Record per-episode displacement and terrain `move_up/move_down` decisions to
   distinguish terrain-curriculum issues from policy movement failure.
5. Compare at iterations 5000, 6500, and 8000 to separate convergence speed
   from final performance.
6. Diagnose B0's post-warmup policy-standard-deviation explosion by adding a
   bounded action standard deviation and logging actor entropy before the next
   full-method run.
