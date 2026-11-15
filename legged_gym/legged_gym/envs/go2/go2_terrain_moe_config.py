from .go2_config import Go2RoughCfg, Go2RoughCfgPPO


class Go2TerrainMoeCfg(Go2RoughCfg):
    class env(Go2RoughCfg.env):
        num_obs_lens = 20
        num_observations = (
            Go2RoughCfg.env.num_one_step_observations * num_obs_lens
        )
        # The migrated tcn-gate checkpoint was trained with base velocity,
        # disturbance, 187 terrain heights, and the 4-D contact mask.
        num_one_step_privileged_obs = 45 + 3 + 3 + 187 + 4
        num_privileged_obs = num_one_step_privileged_obs


class Go2TerrainMoeCfgPPO(Go2RoughCfgPPO):
    class policy(Go2RoughCfgPPO.policy):
        num_modes = 4
        latent_dim = 32
        expert_hidden_dims = (256, 128, 64)
        actor_hidden_dims = (512, 256, 128)
        critic_hidden_dims = (512, 256, 128)
        gate_hidden_dims = (64, 64)
        dynamics_target_dim = 30
        contact_dim = 4
        future_contact_steps = 15
        terrain_dim = 187
        activation = "elu"
        history_encoder = "mlp"
        history_encoder_hidden_dims = (64, 64, 64)
        init_noise_std = 0.8
        ema_tau = 0.01
        gate_rho = 0.3
        gate_type = "mlp"
        gate_tcn_channels = 32
        gate_tcn_layers = 2
        gate_temporal_dim = 32
        gate_kernel_size = 3
        gate_output_norm = False
        gate_balance_coef = 0.0
        terrain_loss_coef = 0.1
        contact_loss_coef = 1.0
        dynamics_loss_coef = 1.0
        kl_loss_coef = 0.1
        velocity_loss_coef = 1.0

    class algorithm(Go2RoughCfgPPO.algorithm):
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        perception_learning_rate = 3.0e-4
        schedule = "adaptive"
        desired_kl = 0.01
        gamma = 0.99
        lam = 0.95

    class runner(Go2RoughCfgPPO.runner):
        runner_class_name = "TerrainMoeRunner"
        policy_class_name = "TerrainMoeActorCritic"
        algorithm_class_name = "TerrainMoePPO"
        empirical_normalization = False
        num_steps_per_env = 24
        max_iterations = 15000
        save_interval = 500
        experiment_name = "go2_terrain_moe"
        run_name = "k4_predictive_ema_gate"


class Go2TerrainMoeSmallCfg(Go2TerrainMoeCfg):
    pass


class Go2TerrainMoeSmallCfgPPO(Go2TerrainMoeCfgPPO):
    class policy(Go2TerrainMoeCfgPPO.policy):
        latent_dim = 16
        history_encoder_hidden_dims = (48, 48, 48)
        tcn_width = 64
        tcn_layers = 3
        expert_hidden_dims = (128, 64, 32)
        actor_hidden_dims = (256, 128, 64)
        critic_hidden_dims = (256, 128, 64)
        gate_hidden_dims = (32,)
        velocity_hidden_dims = (64, 32)
        adapter_hidden_dims = (32,)

    class runner(Go2TerrainMoeCfgPPO.runner):
        experiment_name = "go2_terrain_moe_small"
        run_name = "k4_small_predictive_ema_gate"


class Go2TerrainMoeAttentionCfg(Go2TerrainMoeCfg):
    pass


class Go2TerrainMoeAttentionCfgPPO(Go2TerrainMoeCfgPPO):
    class policy(Go2TerrainMoeCfgPPO.policy):
        num_attn_heads = 4
        route_uniform_floor = 0.1
        attention_temperature = 1.0
        fused_loss_coef = 1.0

    class runner(Go2TerrainMoeCfgPPO.runner):
        policy_class_name = "TerrainMoeAttentionActorCritic"
        experiment_name = "go2_terrain_moe_attn"
        run_name = "k4_mlp_attention"


class Go2TerrainMoeTemporalGateCfg(Go2TerrainMoeCfg):
    pass


class Go2TerrainMoeTemporalGateCfgPPO(Go2TerrainMoeCfgPPO):
    class policy(Go2TerrainMoeCfgPPO.policy):
        gate_type = "temporal_tcn"
        gate_tcn_channels = 32
        gate_tcn_layers = 2
        gate_temporal_dim = 32
        gate_kernel_size = 3
        gate_output_norm = False
        gate_balance_coef = 0.0

    class runner(Go2TerrainMoeCfgPPO.runner):
        policy_class_name = "TerrainMoeActorCritic"
        experiment_name = "go2_terrain_moe_tcn_gate"
        run_name = "k4_mlp_experts_temporal_tcn_gate"


class Go2TerrainMoeTemporalGateV2Cfg(Go2TerrainMoeCfg):
    pass


class Go2TerrainMoeTemporalGateV2CfgPPO(Go2TerrainMoeCfgPPO):
    class policy(Go2TerrainMoeCfgPPO.policy):
        gate_type = "temporal_tcn"
        gate_tcn_channels = 32
        gate_tcn_layers = 3
        gate_temporal_dim = 32
        gate_kernel_size = 3
        gate_output_norm = True
        gate_balance_coef = 0.1

    class runner(Go2TerrainMoeCfgPPO.runner):
        policy_class_name = "TerrainMoeActorCritic"
        experiment_name = "go2_terrain_moe_tcn_gate_v2"
        run_name = "k4_temporal_tcn_norm3_balanced"


class Go2TerrainMoeUniformRoutingCfg(Go2TerrainMoeCfg):
    pass


class Go2TerrainMoeUniformRoutingCfgPPO(Go2TerrainMoeCfgPPO):
    class policy(Go2TerrainMoeCfgPPO.policy):
        gate_type = "uniform"
        gate_balance_coef = 0.0

    class runner(Go2TerrainMoeCfgPPO.runner):
        policy_class_name = "TerrainMoeActorCritic"
        experiment_name = "go2_terrain_moe_uniform_routing"
        run_name = "k4_uniform_routing"


class Go2TerrainMoePPOOnlyCfg(Go2TerrainMoeCfg):
    pass


class Go2TerrainMoePPOOnlyCfgPPO(Go2TerrainMoeTemporalGateV2CfgPPO):
    class policy(Go2TerrainMoeTemporalGateV2CfgPPO.policy):
        use_auxiliary_loss = False
        train_perception_with_ppo = True

    class runner(Go2TerrainMoeTemporalGateV2CfgPPO.runner):
        policy_class_name = "TerrainMoeActorCritic"
        experiment_name = "go2_terrain_moe_ppo_only"
        run_name = "k4_tcn_gate_v2_ppo_only"


class Go2TerrainMoeNoEMACfg(Go2TerrainMoeCfg):
    pass


class Go2TerrainMoeNoEMACfgPPO(Go2TerrainMoeTemporalGateV2CfgPPO):
    class policy(Go2TerrainMoeTemporalGateV2CfgPPO.policy):
        use_ema = False

    class runner(Go2TerrainMoeTemporalGateV2CfgPPO.runner):
        policy_class_name = "TerrainMoeActorCritic"
        experiment_name = "go2_terrain_moe_no_ema"
        run_name = "k4_tcn_gate_v2_no_ema"


class Go2ActorMoECfg(Go2TerrainMoeCfg):
    pass


class Go2ActorMoECfgPPO(Go2TerrainMoeCfgPPO):
    class policy(Go2TerrainMoeCfgPPO.policy):
        num_modes = 4
        actor_hidden_dims = (512, 256, 128)
        critic_hidden_dims = (512, 256, 128)
        gate_hidden_dims = (128, 64)
        velocity_hidden_dims = (128, 64)

    class runner(Go2TerrainMoeCfgPPO.runner):
        policy_class_name = "TerrainActorMoEActorCritic"
        experiment_name = "go2_actor_moe"
        run_name = "k4_actor_moe"
