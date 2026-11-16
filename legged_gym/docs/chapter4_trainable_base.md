# Chapter 4: Trainable Terrain-MoE Base

The default pretrained base is
`go2_terrain_moe_tcn_gate_v2/model_14999.pt`, using three causal TCN gate
layers, LayerNorm gate output, and expert load balancing.

## Control Architecture

```text
Chapter 3 checkpoint (strict validated)
        ↓
Terrain-MoE actor and perception modules
        ↓ partial PPO fine-tuning
12-D Go2 action
        ↓
PD joint targets

Arm IK trajectory curriculum
        ↓
fixed 6-D joint trajectory + target velocity
        ↓
arm PD controller (load disturbance on the base)
```

The arm is no longer a PPO action. Chapter 4 PPO controls only the Go2 base.
Arm commands come from `ArmTrajectoryCurriculum`, IK, and fixed PD control.

## Dimensions

```text
base observation:       45 x 20 = 900
privileged observation: 45 status + 3 linear velocity + 3 disturbance
                        + 187 height + 9 payload + 4 contact = 251
base action:            12
value heads:            3
payload estimate:       9
load fusion weights:    3
```

## Partial Fine-Tuning

The default is `pretrained_policy_finetune_mode=partial`:

- frozen: online predictive experts, EMA target experts, velocity estimator;
- trainable: actor, causal TCN gate, control adapters;
- new and trainable: three critics, payload estimator, load-weight network.

`full` mode can unfreeze online experts and the velocity estimator through
`expert_freeze=False`; EMA target experts always remain gradient-free.

The pretrained checkpoint is first loaded into the original single-critic
network with `strict=True`. Its actor and perception parameters are then copied
into the new trainable model after explicit key/shape checks. The new Chapter 4
critic, payload estimator, and fusion gate are initialized fresh.

## Reward Ownership

Existing coefficients are retained where the corresponding Go2 objective
already existed. Arm tracking rewards are no longer PPO objectives.

| Group | Reward terms |
|---|---|
| task | Go2 linear and angular velocity tracking |
| stability | vertical velocity, roll/pitch rate, orientation, base height, foot slip |
| energy | Go2 joint power, joint acceleration, action rate, smoothness |

The arm still produces rewards only as diagnostics or curriculum signals; it is
not controlled by the PPO actor.

## Arm Trajectory Curriculum

`ArmTrajectoryCurriculum` implements four stages:

1. local arm configuration changes;
2. medium forward/lateral extension;
3. large multi-directional motion;
4. full IK reachable-space sampling.

Promotion requires minimum PPO iterations, trajectory success rate, and base
stability rate. Load difficulty uses changes in COM position, gravity torque,
and gravity-torque rate from the existing normalized payload state.

Settle counting only begins after the local command trajectory is finished.
Curriculum stages, attempt counters, load difficulty, and IK statistics are
saved in Chapter 4 checkpoints.

## Dynamic Advantage Fusion

The normalized payload estimate generates:

```text
lambda = LoadAwareWeightNetwork(payload_estimate)
A_total = sum_i lambda_i * normalized_GAE_i
```

Rollout-time weights are stored and reused during PPO epochs. Actor gradients
cannot update the load gate; gate training uses physical-prior KL supervision
and a masked adjacent-step smoothing loss.

## Commands

Verify the pretrained checkpoint:

```bash
python legged_gym/scripts/verify_base_checkpoint.py
```

Run the short end-to-end pipeline:

```bash
python legged_gym/scripts/test_chapter4_pipeline.py
```

Run unit tests:

```bash
python -m unittest discover -s rsl_rl/rsl_rl/tests -p 'test_*.py' -v
python -m unittest discover \
  -s legged_gym/legged_gym/tests \
  -p 'test_arm_curriculum.py' -v
```

Train:

```bash
GO2_ARM_EXPERIMENT=D python legged_gym/legged_gym/scripts/train.py
```

Run the final paper ablations B0-B4 on two GPUs:

```bash
python legged_gym/legged_gym/scripts/queue_chapter4_ablation.py \
  --gpus 0,1 --num-envs 4096 --seeds 0 \
  --experiment-name chapter4_ablation

python legged_gym/legged_gym/scripts/queue_chapter4_ablation.py \
  --status
```

The queue stores stdout logs, TensorBoard runs, checkpoints, and `manifest.json`
under `legged_gym/logs/chapter4_ablation_queue/` by default.

## Known Limits

- The physical-prior weight parameters remain initial heuristics.
- The 100-update smoke test is not a substitute for full multi-environment
  training or terrain/load generalization evaluation.
- Expert gate usage should be monitored for collapse in long runs.
- MuJoCo and hardware transfer have not been evaluated in this task.
