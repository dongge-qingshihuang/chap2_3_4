import unittest
from pathlib import Path

import torch

from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import TerrainMoeActorCritic
from rsl_rl.runners import HIMOnPolicyRunner
from rsl_rl.storage import HIMRolloutStorage


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = (
    REPO_ROOT
    / "legged_gym"
    / "base_policies"
    / "go2_terrain_moe_tcn_gate_v2"
    / "model_14999.pt"
)


def pretrained_kwargs():
    return dict(
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


class Chapter4PolicyTest(unittest.TestCase):
    def build_trainable_model(self):
        return TerrainMoeActorCritic(
            900,
            251,
            12,
            45,
            None,
            20,
            **pretrained_kwargs(),
            num_critics=3,
            payload_dim=9,
            payload_hidden_dims=(256, 128),
            load_aware_enabled=True,
        )

    def test_pretrained_checkpoint_strict_load(self):
        self.assertTrue(CHECKPOINT.is_file())
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
        model = TerrainMoeActorCritic(
            900,
            242,
            12,
            45,
            None,
            20,
            **pretrained_kwargs()
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        with torch.no_grad():
            actions = model.act_inference(torch.zeros(2, 900))
        self.assertEqual(tuple(actions.shape), (2, 12))
        self.assertTrue(torch.isfinite(actions).all())

    def test_bounded_log_std_and_legacy_std_checkpoint(self):
        model = self.build_trainable_model()
        self.assertNotIn(
            "base_action_residual_enabled",
            model.architecture_metadata(),
        )
        state = model.state_dict()
        self.assertIn("log_std", state)
        self.assertNotIn("std", state)
        self.assertLessEqual(model.std.max().item(), 0.5)
        self.assertGreaterEqual(model.std.min().item(), 0.05)

        with torch.no_grad():
            model.log_std.fill_(2.0)
        clamp_fraction = model.enforce_std_bounds()
        self.assertEqual(clamp_fraction.item(), 1.0)
        self.assertTrue(
            torch.allclose(
                model.std,
                torch.full_like(model.std, 0.5),
            )
        )

        legacy_state = model.state_dict()
        legacy_state.pop("log_std")
        legacy_state["std"] = torch.tensor(
            [0.001] + [2.0] * 11
        )
        model.load_state_dict(legacy_state, strict=True)
        self.assertAlmostEqual(model.std[0].item(), 0.05, places=6)
        self.assertTrue(
            torch.allclose(model.std[1:], torch.full((11,), 0.5))
        )

    def test_partial_freeze_and_gradients(self):
        model = self.build_trainable_model()
        model.configure_finetune(
            "partial",
            expert_freeze=True,
            gate_lr_multiplier=0.25,
            adapter_lr_multiplier=0.25,
            actor_lr_multiplier=1.0,
            expert_lr_multiplier=0.1,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.TsDyn_modules.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.target_TsDyn_modules.parameters()
            )
        )
        self.assertTrue(model.actor[0].weight.requires_grad)
        self.assertTrue(model.control_gate[0].weight.requires_grad)
        self.assertTrue(
            model.control_adapters[0][0].weight.requires_grad
        )
        self.assertTrue(model.critic_heads[0].weight.requires_grad)
        self.assertTrue(model.payload_estimator[0].weight.requires_grad)
        self.assertTrue(model.load_weight_net.net[0].weight.requires_grad)

        model.zero_grad(set_to_none=True)
        observations = torch.randn(4, 900)
        actions = model.act(observations)
        model.get_actions_log_prob(actions).mean().backward()
        self.assertIsNotNone(model.actor[0].weight.grad)
        self.assertIsNotNone(model.control_gate[0].weight.grad)
        self.assertIsNotNone(
            model.control_adapters[0][0].weight.grad
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.TsDyn_modules.parameters()
            )
        )
        self.assertEqual(
            tuple(model.evaluate(torch.randn(4, 251)).shape), (4, 3)
        )

    def test_arm_reference_residual_is_zero_initialized(self):
        model = TerrainMoeActorCritic(
            900,
            251,
            12,
            45,
            None,
            20,
            **pretrained_kwargs(),
            num_critics=3,
            payload_dim=9,
            payload_hidden_dims=(256, 128),
            load_aware_enabled=False,
            base_action_residual_enabled=True,
            base_action_residual_context_dim=18,
            base_action_residual_hidden_dims=(32, 16),
            base_action_residual_clip=0.1,
        )
        observations = torch.randn(4, 900)
        context = torch.randn(4, 18)
        first_residual_layer = next(
            layer
            for layer in model.base_action_residual.modules()
            if isinstance(layer, torch.nn.Linear)
        )
        self.assertEqual(first_residual_layer.in_features, 80 + 18)
        with torch.no_grad():
            mean, _ = model._mean(
                observations, residual_context=context
            )
        self.assertTrue(
            torch.allclose(mean, model.last_base_action_mean, atol=1e-7)
        )
        self.assertTrue(
            torch.equal(
                model.last_action_residual,
                torch.zeros(4, 12),
            )
        )

        with torch.no_grad():
            next(
                layer
                for layer in reversed(
                    list(model.base_action_residual.modules())
                )
                if isinstance(layer, torch.nn.Linear)
            ).bias.fill_(1.0)
            mean, _ = model._mean(
                observations, residual_context=context
            )
        residual = mean - model.last_base_action_mean
        self.assertGreater(residual.abs().max().item(), 0.0)
        self.assertLessEqual(
            residual.abs().max().item(),
            model.base_action_residual_clip + 1e-6,
        )

    def test_residual_only_training_keeps_chapter3_actor_frozen(self):
        model = TerrainMoeActorCritic(
            900,
            251,
            12,
            45,
            None,
            20,
            **pretrained_kwargs(),
            num_critics=3,
            payload_dim=9,
            payload_hidden_dims=(256, 128),
            load_aware_enabled=False,
            base_action_residual_enabled=True,
            base_action_residual_context_dim=18,
            base_action_residual_hidden_dims=(32, 16),
        )
        model.configure_finetune("partial", freeze_actor=True)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.actor.parameters()
            )
        )
        self.assertFalse(model.log_std.requires_grad)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.control_adapters.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.base_action_residual.parameters()
            )
        )
        self.assertTrue(model.critic_heads[0].weight.requires_grad)
        self.assertTrue(model.payload_estimator[0].weight.requires_grad)

        model.zero_grad(set_to_none=True)
        actions = model.act(
            torch.randn(4, 900),
            residual_context=torch.randn(4, 18),
        )
        model.get_actions_log_prob(actions).mean().backward()
        self.assertIsNone(model.actor[0].weight.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.base_action_residual.parameters()
            )
        )

    def test_full_mode_can_unfreeze_experts_and_ema_stays_frozen(self):
        model = self.build_trainable_model()
        model.configure_finetune("full", expert_freeze=False)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.TsDyn_modules.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.target_TsDyn_modules.parameters()
            )
        )
        model.zero_grad(set_to_none=True)
        actions = model.act(torch.randn(2, 900))
        model.get_actions_log_prob(actions).mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.TsDyn_modules.parameters()
            )
        )
        before = [
            parameter.detach().clone()
            for parameter in model.target_TsDyn_modules.parameters()
        ]
        online = [
            parameter.detach().clone()
            for parameter in model.TsDyn_modules.parameters()
        ]
        model.update_ema_target()
        after = [
            parameter.detach().clone()
            for parameter in model.target_TsDyn_modules.parameters()
        ]
        for old, source, new in zip(before, online, after):
            expected = (1.0 - model.ema_tau) * old + model.ema_tau * source
            self.assertTrue(torch.allclose(new, expected, atol=1e-7))

    def test_optimizer_groups_and_deployment_inference(self):
        model = self.build_trainable_model().configure_finetune("partial")
        algorithm = HIMPPO(
            model,
            device="cpu",
            value_dim=3,
            load_aware_enabled=True,
            payload_loss_coef=1.0,
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
        self.assertIn(id(model.log_std), ppo_ids)
        self.assertNotIn(id(model.payload_estimator[0].weight), ppo_ids)
        self.assertIn(id(model.payload_estimator[0].weight), payload_ids)
        self.assertNotIn(id(model.load_weight_net.net[0].weight), ppo_ids)
        self.assertIn(id(model.load_weight_net.net[0].weight), gate_ids)
        self.assertFalse(ppo_ids & payload_ids)
        self.assertFalse(ppo_ids & gate_ids)
        with torch.no_grad():
            actions = model.act_inference(torch.zeros(2, 900))
        self.assertEqual(tuple(actions.shape), (2, 12))
        self.assertEqual(tuple(model.get_payload_estimate().shape), (2, 9))
        self.assertEqual(tuple(model.get_load_weights().shape), (2, 3))

    def test_ppo_act_keeps_pre_action_observation_copy(self):
        model = self.build_trainable_model().configure_finetune("partial")
        algorithm = HIMPPO(
            model,
            device="cpu",
            value_dim=3,
            load_aware_enabled=True,
        )
        observations = torch.randn(2, 900)
        critic_observations = torch.randn(2, 251)
        algorithm.act(observations, critic_observations)
        stored_observations = algorithm.transition.observations
        stored_critic = algorithm.transition.critic_observations
        expected = observations.clone()
        expected_critic = critic_observations.clone()
        observations.zero_()
        critic_observations.zero_()
        self.assertTrue(torch.equal(stored_observations, expected))
        self.assertTrue(torch.equal(stored_critic, expected_critic))
        self.assertNotEqual(
            observations.data_ptr(), stored_observations.data_ptr()
        )


class TimeoutBootstrapTest(unittest.TestCase):
    def test_timeout_uses_terminal_next_value_and_next_step_is_normal(self):
        storage = HIMRolloutStorage(
            1, 2, [1], [1], [1], value_dim=1
        )
        storage.rewards[:, 0, 0] = torch.tensor([1.0, 2.0])
        storage.values[:, 0, 0] = torch.zeros(2)
        storage.dones[:, 0, 0] = torch.tensor([1, 0], dtype=torch.uint8)
        storage.timeouts[:, 0] = torch.tensor([True, False])
        storage.timeout_bootstrap_values[:, 0, 0] = torch.tensor([5.0, 0.0])
        storage.compute_returns(
            torch.zeros(1, 1), gamma=0.9, lam=0.8
        )
        expected_step1 = 2.0
        expected_step0 = (
            1.0
            + 0.9 * 5.0
            + 0.9 * 0.8 * expected_step1
        )
        self.assertAlmostEqual(
            storage.returns[0, 0, 0].item(), expected_step0, places=6
        )
        self.assertAlmostEqual(
            storage.returns[1, 0, 0].item(), expected_step1, places=6
        )

    def test_fall_does_not_bootstrap(self):
        storage = HIMRolloutStorage(
            1, 1, [1], [1], [1], value_dim=1
        )
        storage.rewards[0, 0, 0] = 1.0
        storage.values[0, 0, 0] = 0.0
        storage.dones[0, 0, 0] = 1
        storage.timeouts[0, 0] = False
        storage.timeout_bootstrap_values[0, 0, 0] = 99.0
        storage.compute_returns(
            torch.zeros(1, 1), gamma=0.9, lam=0.8
        )
        self.assertAlmostEqual(
            storage.returns[0, 0, 0].item(), 1.0, places=6
        )


class EpisodeAggregationTest(unittest.TestCase):
    def test_episode_batches_are_weighted_by_completed_count(self):
        metrics = HIMOnPolicyRunner._aggregate_episode_infos(
            [
                {
                    "_episode_count": 1,
                    "rew_task": torch.tensor(2.0),
                },
                {
                    "_episode_count": 99,
                    "rew_task": torch.tensor(4.0),
                },
            ]
        )
        self.assertAlmostEqual(
            metrics["rew_task"].item(), 3.98, places=6
        )
        self.assertNotIn("_episode_count", metrics)


if __name__ == "__main__":
    unittest.main()
