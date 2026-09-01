# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch

# =============================================================================
# 章节开关：切换第三章 / 第四章的训练配置（改这一行即可）
#   "rl"          第三章：纯学习端到端伺服，动作 q = q_current + Δq·action_scale
#   "ik_residual" 第四章：模型-学习混合，动作 q = q_ik + Δq·residual_scale
# =============================================================================
ARM_CONTROL_MODE = "ik_residual"

# 第四章下是否把解析逆解 q_ik 相对当前构型的偏差写进观测（消融用）
INCLUDE_IK_IN_OBS = True

# 由上面两个开关派生：观测中 IK 块的维度（第三章恒为 0，保证观测布局与原来一致）
_IK_OBS_DIM = 6 if (ARM_CONTROL_MODE == "ik_residual" and INCLUDE_IK_IN_OBS) else 0

# =============================================================================
# 第二章开关：底座对机械臂负载的感知方式
#   "none"      不感知（消融基线）：底座观测保持 45 维纯本体量
#   "estimated" 黑盒估计（主线）：本体观测不变，由 ProprioEncoder 从本体历史估计负载状态，
#               解析真值走特权通道做监督。部署时只需本体感知，不依赖臂编码器
#   "explicit"  白盒前馈（上界参照）：把解析负载状态直接拼进底座观测
# 负载状态 9 维 = 等效质心位置(3) + 重力力矩(3) + 质心速度(3)，均在 base 系
# =============================================================================
PAYLOAD_AWARENESS = "estimated"

_PAYLOAD_DIM = 9
_PAYLOAD_OBS_DIM = _PAYLOAD_DIM if PAYLOAD_AWARENESS == "explicit" else 0
_PAYLOAD_PRIV_DIM = 0 if PAYLOAD_AWARENESS == "none" else _PAYLOAD_DIM

# =============================================================================
# 第二章开关：值函数分解（多评论家）
# 开启后奖励按 task / stab / energy 三组分别累加、critic 输出三个头、各自独立做 GAE，
# 最后按 λ 加权成标量 advantage 驱动策略更新。
# 奖励项通过 "组名__函数名" 的命名归组（如 stab__orientation）；无前缀的一律算 task，
# 因此第三、四章现有的奖励配置无需改动。
# =============================================================================
MULTI_CRITIC = False

_VALUE_DIM = 3 if MULTI_CRITIC else 1
REWARD_GROUPS = ("task", "stab", "energy")


class Go2ArmRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'widow_waist': 0., 
            'widow_shoulder': 0., 
            'widow_elbow': 0., 
            'forearm_roll': 0., 
            'widow_wrist_angle': 0., 
            'widow_wrist_rotate': 0., 
            'widow_forearm_roll': 0., 
            'gripper': 0., 
            'widow_left_finger': 0., 
            'widow_right_finger': 0.,
            
            "zarx_j1": 0.0,
            "zarx_j2": 0.8,
            "zarx_j3": 0.8,
            "zarx_j4": 0.0,
            "zarx_j5": 0.0,
            "zarx_j6": 0.0,
            "zarx_j7": 0.0,
            "zarx_j8": 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        stiffness_arm = {
            "zarx": 50,
            "zarx_j1": 10,
            "zarx_j2": 70,
            "zarx_j3": 70,
            "zarx_j4": 25,
            "zarx_j5": 25,
            "zarx_j6": 25,
            "zarx_j7": 50,
            "zarx_j8": 50,
    }  # [N*m/rad]
        damping_arm = {
                "zarx": 20,
                "zarx_j1": 3,
                "zarx_j2": 15,
                "zarx_j3": 15,
                "zarx_j4": 2,
                "zarx_j5": 2,
                "zarx_j6": 2,
                "zarx_j7": 20,
                "zarx_j8": 20,
        }  # [N*m*s/rad]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
        keep_arm_fixed = False  

    class arm:
        ik_lib_path = "/workspace/loco-manipulation/legged_gym/legged_gym/envs/go2_arm/ee_lib_link6_0.6.npz"
        
        curriculum_iters = [2000, 4000, 8000]  # 课程切换的迭代点


        base_height_offset = 0.38

        class joint_finetune:
            # 底座顺应残差（Phase 1 联合 fine-tune）：冻结底座照旧走路，
            # 额外训一个残差网络让底座主动下蹲/前倾/挪重心去帮机械臂够到目标
            # 已验证：在不可达命令下残差无收益。先关掉，待命令改为可达后再评估
            enabled = False
            residual_lr = 5e-4          # 残差独立的低学习率
            residual_clip = 1         # 残差动作限幅 [rad]（叠加在 dog 动作上）
            residual_entropy_coef = 0.002
            # 残差 entropy 退火：从头开始线性退火到 0.0005。
            # 上一版 std 在 it825 就顶满 0.5、退火 it2000 才启动完全没用上，
            # 这里直接 start_frac=0 从头压
            residual_entropy_coef_final = 0.0005
            residual_entropy_anneal_start_frac = 0.0
            init_noise_std = 0.1
            std_max = 0.25              # 直接砍掉一半探索噪声上限（旧值 0.5 会被顶满）
            hidden_dims = (256, 256)
            roll_term = 1.2             # 放宽 roll 终止阈值（默认 0.8）
            pitch_term = 1.2            # 放宽 pitch 终止阈值（默认 0.8）

        class settle:
            # 事件驱动 "到达-保持-换目标"
            pos_tol = 0.06        # [m] EE 位置误差容差（课程初始值）
            ori_tol = 0.25        # [rad] EE 姿态误差容差（课程初始值）
            hold_steps = 15       # 连续保持步数 (0.5s @ 50Hz) 才算 settle
            # PBRS shaping: phi = -(pos_err + pbrs_ori_weight * ori_err)
            pbrs_gamma = 0.99     # 与 PPO gamma 保持一致
            pbrs_ori_weight = 0.4  # 轻微偏向姿态：0.5 过头导致位置被饿，回调到 0.4

            # 容差课程（self-paced）：位置/姿态按各自轴的"在容差内时间占比"独立收紧，
            # 避免姿态这个较难的轴拖累位置轴一起停滞
            curriculum = True
            pos_tol_final = 0.03       # [m] 最终精度
            ori_tol_final = 0.08        # [rad] 最终精度
            tol_decay = 0.9             # 每次收紧的系数
            pos_intol_target = 0.25     # 位置在容差内时间占比超过该值则收紧位置容差
            ori_intol_target = 0.25     # 姿态在容差内时间占比超过该值则收紧姿态容差
            settle_rate_target = 0.04   # 保留：settle 事件率（两轴同时达标）参考用
            curriculum_interval = 50    # 每隔多少个 PPO 迭代评估一次

        class commands:
            # 命令来源（保证位置+姿态联合可达，解决 both_success 仅 1.4% 的根因）：
            #   'ik_lib_aug'  —— 以 IK 库为骨架 + 库内插值 + 局部扰动（推荐：可达、朝向自然、覆盖比纯库宽）
            #   'ik_lib'      —— 纯 IK 库（朝向多样性差，每个位置仅一种朝向）
            #   'joint_space' —— 关节全 limit 均匀采样（覆盖最全但含病态极端腕态，姿态过难）
            #   'random'      —— 旧独立随机采样（位姿大多不可达，仅作对照）
            # 非 'random' 时，下面 l/p/y/rpy 范围不再使用（命令来自 q 的 FK）
            command_source = "ik_lib_aug"
            joint_sample_l_range = [0.3, 0.65]   # FK 末端距离接受窗口 [m]（库增广/正向采样都用）
            ik_interp_alpha_max = 0.5            # 库内两配置插值系数上限
            # 库配置局部扰动 std [rad]，按关节分层：近端(j1~j3)小(主要动位置)，
            # 腕部(j4~j6)大(主要动姿态)，以补齐"给定位置下的姿态多样性"。
            # 标量则各关节相同。顺序 zarx_j1..j6。
            ik_perturb_std = [0.05, 0.08, 0.08, 0.40, 0.50, 0.60]
            ik_lib_path = "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/go2_arm/ee_lib_link6.npz"

            # 躯干/髋部包围盒（base 系 [m]）：末端命令落入该盒视为自碰撞，拒绝重采
            body_reject = True
            body_box_x = [-0.25, 0.30]
            body_box_y = [-0.12, 0.12]
            body_box_z = [-0.15, 0.12]

            # 局部轨迹跟踪：重采新目标时命令从当前位姿平滑过渡到新目标（smoothstep），
            # 而非瞬间跳变。reset 例外（直接给静态目标）。
            class trajectory:
                enabled = True
                duration_s = 1.0      # 起点到终点的过渡时长 [s]

            l = [0.3, 0.7]
            p = [-torch.pi*0.4 , torch.pi*0.4]  # 75 
            y = [-torch.pi/2 , torch.pi/2]
            roll_ee = [-torch.pi * 0.5, torch.pi * 0.5]
            pitch_ee = [-torch.pi/3*1.1 , torch.pi/3*1.1]
            yaw_ee = [-1.3 , 1.3]

        class ik_control:
            # 第四章：解析逆运动学骨架 + 强化学习残差。mode 由文件顶部的 ARM_CONTROL_MODE 控制
            mode = ARM_CONTROL_MODE          # "rl"(第三章) | "ik_residual"(第四章)
            include_ik_in_obs = INCLUDE_IK_IN_OBS
            obs_dim = _IK_OBS_DIM

            # 阻尼最小二乘迭代逆解：Δq = Jᵀ(JJᵀ + λ²I)⁻¹ e
            iters = 4                        # 每个控制步的迭代次数（带热启动，少量即可）
            damping = 0.05                   # λ，奇异位形附近的阻尼
            step_scale = 0.5                 # 每次迭代的步长缩放
            max_step_norm = 0.3              # [rad] 单次迭代关节增量的范数上限
            ori_weight = 1.0                 # 误差向量中姿态项相对位置项的权重

            # 残差动作缩放。第四章的 Δq 是修正量，尺度应明显小于第三章的 action_scale(0.25)
            residual_scale = 0.10
            residual_clip = 0.5              # [rad] 残差限幅，保证不偏离名义解太远

        class payload:
            # 第二章：底座对机械臂负载的感知。mode 由文件顶部的 PAYLOAD_AWARENESS 控制
            mode = PAYLOAD_AWARENESS         # "none" | "estimated" | "explicit"
            dim = _PAYLOAD_DIM               # 9 = 质心位置(3) + 重力力矩(3) + 质心速度(3)
            obs_dim = _PAYLOAD_OBS_DIM       # 进本体观测的维度（仅 explicit 非零）
            priv_dim = _PAYLOAD_PRIV_DIM     # 进特权观测的维度（estimated/explicit 均为 9）
            # 归一化尺度，让三组量纲相近后再喂网络 / 算监督损失
            com_scale = 0.3                  # [m]
            torque_scale = 10.0              # [N·m]
            com_vel_scale = 1.0              # [m/s]
            estimator_loss_coef = 1.0        # 黑盒估计的监督损失权重

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # dog 45 (+ 第二章白盒负载 9) + arm 30 (+ 第四章 IK 块 6)
        # 两个开关都关时为 75，布局与原来完全一致
        num_dog_one_step_observations = 45 + _PAYLOAD_OBS_DIM
        num_one_step_observations = num_dog_one_step_observations + 30 + _IK_OBS_DIM
        num_hist = 6 #6
        num_observations = num_one_step_observations * num_hist
        num_recon = num_one_step_observations # next one step obs: current_obs without command + command + base_lin_vel
        # = one_step_obs + base_lin_vel(3) + disturbance(3) + height_scan(187)
        #   + 第二章负载真值(9，作监督目标) + contact_mask(4)
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187 + _PAYLOAD_PRIV_DIM + 4
        # 值函数维度：1 = 单评论家（第三/四章），3 = task/stab/energy 三头（第二章）
        value_dim = _VALUE_DIM
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        num_arm_actions = 8

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        flat_warmup_iters = 1000
        flat_warmup_level = 0
        mesh_type = 'trimesh'
        curriculum = True
        # rough terrain only:
        measure_heights = True
        # static_friction = 0.6
        # dynamic_friction = 0.6
        
        static_friction = 0.6
        dynamic_friction = 0.6
        
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 6  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        
        terrain_dict = {"flat": 0.05, 
                        "rough flat": 0.05,
                        "slope up": 0.075,
                        "slope down": 0.075, 
                        "rough slope up": 0.075,
                        "rough slope down": 0.075, 
                        "stairs up": 0.20, 
                        "stairs down": 0.10,
                        "discrete": 0.2}
        
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.4]   # rad
        rough_slope_range = [0.0, 0.05]
        # stair_width_range = [0.50, 0.40]
        stair_width_range = [0.35, 0.30]
        stair_height_range = [0.10, 0.20]
        # stair_height_range = [0.15, 0.20]
        discrete_height_range = [0.05, 0.10]
        restitution = 0.
        fractal_noise_strength = 0.05
        
        # measured_points_x = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]


    class commands( LeggedRobotCfg.commands ):
            curriculum = False
            max_curriculum = 1.5
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-1.0, 1.5] # min max [m/s]
                lin_vel_y = [-1.0, 1.0]   # min max [m/s]
                ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
                heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2_arm/urdf/arx5go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        ee_name = "zarx_body6"
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
        



            arm_manip_commands_tracking_combine = 3

            # 精修项：窄 sigma 的 fine 跟踪奖励，只在接近目标时强烈起作用，
            # 弥补 combine（宽 exp）在目标附近梯度变平、推不动最后几 cm/度的问题。
            # 姿态是瓶颈轴，权重略高。
            arm_pos_tracking_fine = 2.5   # sigma=0.03 m
            arm_ori_tracking_fine = 3.5   # sigma=0.12 rad

            # 事件驱动 settle bonus（一次性稀疏奖励，scale 会乘 dt=0.02，实际单次 +2.0）
            arm_settle = 300
            # PBRS 误差进展 shaping（gamma*phi' - phi，理论上不改变最优策略）
            arm_pbrs_progress = 100
            # 臂动作与底座扰动的耦合惩罚（量级较大，从保守值起步，可逐步加强）
            arm_base_coupling = -0.01

            # 底座顺应残差幅度惩罚。仅在 arm.joint_finetune.enabled=True 时有意义；
            # 残差关闭时 dog_residual 恒为 0，此项恒零，故置 0 避免占用奖励槽与日志
            base_residual_penalty = 0.0

            arm_action_rate = -0.01
            # arm_smoothness = -0.01
            arm_energy = -1e-6
            # arm_dof_vel = 0
            arm_dof_acc = -1e-7

            orientation = -0.2
            # tracking_lin_vel = 0.5
            # tracking_ang_vel = 0.2
            
            






        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.32
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.20
        manip_weight_lpy = 1.0
        manip_weight_rpy = 1.2  # 轻微偏向姿态：1.5 过头导致位置被饿，回调到 1.2
        manip_ori_norm = 0.6    # 姿态测地误差 [rad] 归一化尺度（0.6rad→1.0，与位置项量级可比）

class Go2ArmRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # entropy 退火：前期 0.01 保证探索/课程启动，后期降到 0.002 让 std 下降，
        # 末端噪声安静后才能在容差内 hold 住（前 25% 迭代恒定，之后线性退火）
        entropy_coef = 0.01
        entropy_coef_final = 0.002
        entropy_anneal_start_frac = 0.25
        # 多评论家：三路 advantage 的加权系数（value loss 也用同一组权重）
        value_dim = _VALUE_DIM
        lambda_task = 0.7
        lambda_stab = 0.2
        lambda_energy = 0.1
    class runner( LeggedRobotCfgPPO.runner ):
        empirical_normalization = True
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 8000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = 'baseline'
        use_frozen_base_policy = True
        base_policy_path = "/workspace/manipulation/legged_gym/logs/go2_arm/baseline/model_16500.pt"
        base_empirical_normalization = True
        # 冻结底座 checkpoint 训练时用的 payload_dim（第二章开关）。现有的
        # model_16500.pt 是无负载感知时训的，故为 0；换用带负载感知的第二章新权重后改成 9
        base_policy_payload_dim = 0

  