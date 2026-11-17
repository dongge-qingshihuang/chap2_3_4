# Chapter 4 Ablation Training Report

Snapshot time: 2026-11-17 09:54 Asia/Shanghai

> **Formal run status:** B0 and B1 completed all 8000 updates. B4 and B2 are
> still running; B3 and B0F are pending. B1 is healthy, while B0 shows a severe
> late-training collapse: action noise rose above 10, mean reward dropped to a
> transient minimum below -600, and final task tracking/terrain progress fell
> far behind B1. Numerical results from earlier provisional queues are retained
> only as debugging baselines.
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

## Queue Status

| Mode | Description | Status |
|---|---|---|
| B0 | Full method (`Ours`) | Finished, late-training collapse |
| B1 | Single critic, no load prior | Finished, healthy |
| B4 | No pretrained initialization | Running, intermediate |
| B2 | No arm trajectory curriculum | Running, intermediate |
| B3 | No load-aware fusion, fixed weights | Pending |
| B0F | Pretrained control with B4's full unfreeze scope | Pending |

## Formal Training Snapshot

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
