import unittest

import torch

from rsl_rl.modules import (
    Actor_mulCritic_attention1,
    LoadAwareWeightNetwork,
    physical_prior_weights,
)
from rsl_rl.algorithms import HIMPPO
from rsl_rl.storage import HIMRolloutStorage


def build_small_actor(load_aware=True):
    return Actor_mulCritic_attention1(
        num_actor_obs=12,
        num_critic_obs=10,
        num_actions=3,
        num_proprio=2,
        num_recon=2,
        history_len=6,
        num_modes=2,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        tsdyn_hidden_dims=(8, 4),
        tsdyn_latent_dims=4,
        terrain_dim=4,
        num_attn_heads=2,
        payload_dim=9,
        load_aware_enabled=load_aware,
    )


class LoadAwareWeightTest(unittest.TestCase):
    def test_simplex_and_configured_bounds(self):
        network = LoadAwareWeightNetwork()
        payload = torch.randn(32, 9) * 3.0
        weights = network(payload)
        self.assertEqual(tuple(weights.shape), (32, 3))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0.05 - 1e-6).all())
        self.assertTrue((weights <= 0.90 + 1e-6).all())
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=-1),
                torch.ones(32),
                atol=1e-5,
            )
        )

    def test_physical_prior_reacts_to_load(self):
        low = physical_prior_weights(torch.zeros(1, 9))["weights"]
        high = physical_prior_weights(torch.full((1, 9), 3.0))["weights"]
        self.assertGreater(high[0, 1], low[0, 1])
        self.assertTrue(
            torch.allclose(high.sum(dim=-1), torch.ones(1), atol=1e-5)
        )
        expected_ratio = high[0, 0] / (high[0, 0] + high[0, 2])
        self.assertAlmostEqual(expected_ratio.item(), 0.8, places=5)


class MultiCriticGradientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = build_small_actor()
        self.obs = torch.randn(4, 12)
        self.critic_obs = torch.randn(4, 10)

    def _clear_grads(self):
        for parameter in self.model.parameters():
            parameter.grad = None

    def test_three_critic_outputs(self):
        values = self.model.evaluate(self.critic_obs)
        self.assertEqual(tuple(values.shape), (4, 3))

    def test_deployment_inference_uses_proprio_history_only(self):
        actions, _, _ = self.model.act_inference(self.obs)
        self.assertEqual(tuple(actions.shape), (4, 3))
        self.assertEqual(
            tuple(self.model.get_payload_estimate().shape), (4, 9)
        )
        weights = self.model.get_load_weights()
        self.assertEqual(tuple(weights.shape), (4, 3))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=-1),
                torch.ones(4),
                atol=1e-5,
            )
        )

    def test_payload_loss_only_updates_payload_path(self):
        self._clear_grads()
        _, _, payload = self.model.prop_encoder.forward_full(self.obs)
        self.assertEqual(tuple(payload.shape), (4, 9))
        payload.pow(2).mean().backward()
        self.assertIsNotNone(self.model.prop_encoder.payload_head.weight.grad)
        self.assertIsNone(self.model.actor[0].weight.grad)
        self.assertIsNone(self.model.critic_trunk[0].weight.grad)
        self.assertIsNone(
            self.model.load_weight_net.net[0].weight.grad
        )

    def test_gate_loss_only_updates_gate(self):
        self._clear_grads()
        _, _, payload = self.model.prop_encoder.forward_full(self.obs)
        predicted = self.model.load_weight_net(payload.detach())
        prior = physical_prior_weights(
            torch.randn(4, 9)
        )["weights"].detach()
        loss = (
            prior
            * (
                prior.clamp_min(1e-8).log()
                - predicted.clamp_min(1e-8).log()
            )
        ).sum(dim=-1).mean()
        loss.backward()
        self.assertIsNotNone(
            self.model.load_weight_net.net[0].weight.grad
        )
        self.assertIsNone(self.model.actor[0].weight.grad)
        self.assertIsNone(self.model.critic_trunk[0].weight.grad)
        self.assertIsNone(self.model.prop_encoder.payload_head.weight.grad)

    def test_actor_loss_does_not_update_gate_or_critic(self):
        self._clear_grads()
        actions = self.model.act(self.obs)
        self.model.get_actions_log_prob(actions).mean().backward()
        self.assertIsNotNone(self.model.actor[0].weight.grad)
        self.assertIsNone(
            self.model.load_weight_net.net[0].weight.grad
        )
        self.assertIsNone(self.model.critic_trunk[0].weight.grad)

    def test_value_loss_reaches_all_heads(self):
        self._clear_grads()
        values = self.model.evaluate(self.critic_obs)
        values.pow(2).mean().backward()
        self.assertIsNotNone(self.model.critic_trunk[0].weight.grad)
        self.assertIsNotNone(self.model.critic_task.weight.grad)
        self.assertIsNotNone(self.model.critic_stab.weight.grad)
        self.assertIsNotNone(self.model.critic_energy.weight.grad)

    def test_checkpoint_roundtrip_and_metadata(self):
        clone = build_small_actor()
        clone.load_state_dict(self.model.state_dict(), strict=True)
        self.assertEqual(
            self.model.architecture_metadata(),
            clone.architecture_metadata(),
        )
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, clone.state_dict()[key]))
        without_gate = build_small_actor(load_aware=False)
        self.assertNotEqual(
            self.model.architecture_metadata(),
            without_gate.architecture_metadata(),
        )
        with self.assertRaises(RuntimeError):
            without_gate.load_state_dict(
                self.model.state_dict(), strict=True
            )


class RolloutStorageTest(unittest.TestCase):
    def _transition(self, stored, dones, weights):
        transition = HIMRolloutStorage.Transition()
        transition.observations = torch.zeros(1, 1)
        transition.critic_observations = torch.zeros(1, 1)
        transition.next_critic_observations = torch.zeros(1, 1)
        transition.actions = torch.zeros(1, 1)
        transition.rewards = stored.clone()
        transition.dones = dones.clone()
        transition.values = torch.zeros(1, 3)
        transition.actions_log_prob = torch.zeros(1)
        transition.action_mean = torch.zeros(1, 1)
        transition.action_sigma = torch.ones(1, 1)
        transition.payload_predictions = torch.zeros(1, 9)
        transition.load_weights = weights.clone()
        return transition

    def test_independent_gae_and_exact_fusion(self):
        steps, envs, value_dim = 4, 1, 3
        storage = HIMRolloutStorage(
            envs,
            steps,
            [1],
            [1],
            [1],
            value_dim=value_dim,
        )
        rewards = torch.tensor(
            [
                [[1.0, 0.1, -0.2]],
                [[2.0, 0.3, -0.1]],
                [[3.0, 0.7, -0.4]],
                [[4.0, 1.1, -0.8]],
            ]
        )
        values = torch.zeros(steps, envs, value_dim)
        dones = torch.tensor(
            [[[0]], [[1]], [[0]], [[0]]], dtype=torch.bool
        )
        storage.rewards.copy_(rewards)
        storage.values.copy_(values)
        storage.dones.copy_(dones)
        storage.compute_returns(torch.zeros(envs, value_dim), 0.99, 0.95)

        expected_returns = torch.zeros_like(values)
        advantage = torch.zeros(value_dim)
        for step in reversed(range(steps)):
            next_values = (
                torch.zeros(value_dim)
                if step == steps - 1
                else values[step + 1, 0]
            )
            nonterminal = 1.0 - dones[step, 0, 0].float()
            delta = (
                rewards[step, 0]
                + nonterminal * 0.99 * next_values
                - values[step, 0]
            )
            advantage = delta + nonterminal * 0.99 * 0.95 * advantage
            expected_returns[step, 0] = advantage + values[step, 0]

        expected_advantages = expected_returns - values
        expected_advantages = (
            expected_advantages - expected_advantages.mean(dim=(0, 1), keepdim=True)
        ) / (expected_advantages.std(dim=(0, 1), keepdim=True) + 1e-8)
        self.assertTrue(
            torch.allclose(storage.advantages, expected_advantages, atol=1e-6)
        )
        weights = torch.tensor([0.7, 0.2, 0.1]).view(1, 1, 3)
        storage.compute_fused_advantages(weights)
        expected_fused = (
            storage.advantages * weights
        ).sum(dim=-1, keepdim=True)
        self.assertTrue(
            torch.allclose(storage.fused_advantages, expected_fused, atol=1e-7)
        )
        self.assertNotEqual(
            storage.advantages[0, 0, 0],
            storage.advantages[0, 0, 1],
        )

    def test_weight_smoothing_mask_excludes_episode_boundary(self):
        storage = HIMRolloutStorage(
            1, 3, [1], [1], [1], value_dim=3
        )
        weights = torch.tensor([[0.7, 0.2, 0.1]])
        storage.add_transitions(
            self._transition(torch.ones(1, 3), torch.tensor([False]), weights)
        )
        storage.add_transitions(
            self._transition(torch.ones(1, 3), torch.tensor([True]), weights)
        )
        storage.add_transitions(
            self._transition(torch.ones(1, 3), torch.tensor([False]), weights)
        )
        self.assertTrue(
            torch.equal(
                storage.load_weight_smooth_mask[:, 0],
                torch.tensor([[0.0], [1.0], [0.0]]),
            )
        )


class OptimizerOwnershipTest(unittest.TestCase):
    def test_parameter_optimizers_do_not_overlap(self):
        model = build_small_actor()
        algorithm = HIMPPO(
            model,
            device="cpu",
            value_dim=3,
            load_aware_enabled=True,
        )
        ppo_ids = {
            id(parameter)
            for group in algorithm.optimizer.param_groups
            for parameter in group["params"]
        }
        payload_ids = {
            id(parameter)
            for group in algorithm.payload_optimizer.param_groups
            for parameter in group["params"]
        }
        gate_ids = {
            id(parameter)
            for group in algorithm.load_gate_optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertIn(id(model.actor[0].weight), ppo_ids)
        self.assertIn(id(model.critic_task.weight), ppo_ids)
        self.assertNotIn(id(model.prop_encoder.payload_head.weight), ppo_ids)
        self.assertIn(id(model.prop_encoder.payload_head.weight), payload_ids)
        self.assertNotIn(id(model.load_weight_net.net[0].weight), ppo_ids)
        self.assertIn(id(model.load_weight_net.net[0].weight), gate_ids)
        self.assertFalse(ppo_ids & payload_ids)
        self.assertFalse(ppo_ids & gate_ids)
        self.assertFalse(payload_ids & gate_ids)


if __name__ == "__main__":
    unittest.main()
