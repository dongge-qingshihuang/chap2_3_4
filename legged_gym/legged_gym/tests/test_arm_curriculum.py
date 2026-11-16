import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.

import unittest

import torch

from legged_gym.envs.go2_arm.arm_trajectory_curriculum import (
    ArmTrajectoryCurriculum,
)


class FakeArmEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.device = "cpu"
        self.arm_control_dof_slice = slice(0, 6)
        self.dof_pos = torch.randn(num_envs, 8) * 0.2
        self.arm_q_lower = torch.full((6,), -2.0)
        self.arm_q_upper = torch.full((6,), 2.0)
        self.ik_lib_q = torch.randn(32, 6) * 0.2
        self.ik_lib_size = 32
        self.ik_perturb_std = torch.full((1, 6), 0.1)
        self.joint_sample_l_min = 0.1
        self.joint_sample_l_max = 0.8

    def _fk_arm_mount_to_ee(self, q):
        unit = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return unit * 0.5, torch.zeros(q.size(0), 4)

    def _ee_in_body(self, ee):
        return torch.zeros(ee.size(0), dtype=torch.bool)


class ArmTrajectoryCurriculumTest(unittest.TestCase):
    def test_four_stages_sample_and_promote(self):
        env = FakeArmEnv()
        curriculum = ArmTrajectoryCurriculum(
            4,
            "cpu",
            min_iterations=(0, 10, 20, 30),
            evaluation_window=2,
        )
        curriculum.stage[:] = torch.arange(4)
        sample = curriculum.sample_q(env, torch.arange(4))
        self.assertEqual(tuple(sample.shape), (4, 6))
        self.assertTrue(torch.isfinite(sample).all())

        curriculum.stage[:] = 0
        curriculum.attempts[:] = 2
        curriculum.successes[:] = 2
        curriculum.stable_attempts[:] = 2
        curriculum.update_stages(10)
        self.assertTrue(
            torch.equal(
                curriculum.stage,
                torch.ones(4, dtype=torch.long),
            )
        )

    def test_settle_results_ignore_unstarted_attempts(self):
        curriculum = ArmTrajectoryCurriculum(2, "cpu")
        curriculum.record_result(
            torch.tensor([0, 1]), torch.tensor([True, False]), torch.tensor([True, False])
        )
        self.assertTrue(
            torch.equal(
                curriculum.successes,
                torch.zeros(2, dtype=torch.long),
            )
        )
        curriculum.begin_trajectories(torch.tensor([0, 1]))
        curriculum.record_result(
            torch.tensor([0, 1]),
            torch.tensor([True, False]),
            torch.tensor([True, True]),
        )
        self.assertEqual(curriculum.successes.tolist(), [1, 0])
        self.assertEqual(curriculum.stable_attempts.tolist(), [1, 1])

    def test_state_roundtrip(self):
        curriculum = ArmTrajectoryCurriculum(3, "cpu")
        curriculum.stage[:] = torch.tensor([0, 2, 3])
        curriculum.attempts[:] = torch.tensor([5, 6, 7])
        state = curriculum.state_dict()
        restored = ArmTrajectoryCurriculum(3, "cpu")
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.stage, curriculum.stage))
        self.assertTrue(torch.equal(restored.attempts, curriculum.attempts))


if __name__ == "__main__":
    unittest.main()
