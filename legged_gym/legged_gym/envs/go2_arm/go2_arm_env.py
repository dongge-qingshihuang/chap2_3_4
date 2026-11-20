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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from collections import deque

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float,quaternion_to_rpy
from legged_gym.utils.helpers import class_to_dict
from .go2_arm_config import Go2ArmRoughCfg
from .arm_trajectory_curriculum import ArmTrajectoryCurriculum

class Go2ArmEnv(BaseTask):
    def __init__(self, cfg: Go2ArmRoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        # 第二章负载感知模式。_create_envs 会用到，必须在 super().__init__ 之前确定
        self.payload_awareness = getattr(self.cfg.arm.payload, "mode", "none")
        self.payload_obs_dim = int(getattr(self.cfg.arm.payload, "obs_dim", 0))
        self.payload_priv_dim = int(getattr(self.cfg.arm.payload, "priv_dim", 0))
        # 值函数维度：1 = 单评论家（第三/四章），3 = task/stab/energy 三头（第二章）
        self.value_dim = int(getattr(self.cfg.env, "value_dim", 1))
        self.reward_groups = ("task", "stab", "energy")
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.history_length = int(self.num_obs / self.num_one_step_obs)
        self.num_dog_actions = 12              # 狗的 DOF 数，不是 policy action
        self.num_arm_dofs = 8                  # 机械臂总 DOF（含 2 个夹爪）
        self.num_arm_actions = 6               # PD-controlled arm joints
        self.num_actions = 12                  # Chapter 4 PPO controls Go2 only

        self.dog_dof_slice = slice(0, 12)
        self.arm_dof_slice = slice(12, 20)     # arm 全 8 关节
        self.arm_control_dof_slice = slice(12, 18)   # 前 6 个可控关节
        self.gripper_dof_slice = slice(18, 20)       # 后 2 个夹爪固定
        self.keep_arm_fixed = self.cfg.control.keep_arm_fixed
        trajectory_cfg = getattr(
            self.cfg.arm, "trajectory_curriculum", None
        )
        self.arm_fixed_warmup_iters = int(
            getattr(
                trajectory_cfg,
                "fixed_default_warmup_iters",
                0,
            )
        )
        self._arm_warmup_was_active = (
            self.arm_fixed_warmup_iters > 0
        )
        # 第三章 "rl"：q = q_current + Δq·action_scale；第四章 "ik_residual"：q = q_ik + Δq·residual_scale
        self.use_ik_control = getattr(self.cfg.arm.ik_control, "mode", "rl") == "ik_residual"
        self.ik_obs_dim = int(getattr(self.cfg.arm.ik_control, "obs_dim", 0))
        # Physical payload state is also needed by the trajectory curriculum
        # when load-aware fusion itself is disabled (B3).
        self.arm_body_masses = self.default_rigid_body_mass[
            self.arm_body_indices
        ]
        self.arm_body_mass_sum = self.arm_body_masses.sum().clamp(min=1e-6)
        if self.payload_awareness != "none":
            print(f"[Payload] mode={self.payload_awareness} 臂连杆 {self.arm_body_indices.numel()} 个, "
                  f"总质量 {self.arm_body_mass_sum.item():.3f} kg")
        self.effort_flag = 1
        self.ppo_it = 0



        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        # 在 __init__ 末尾（_init_buffers() 之后）
        self._prepare_reward_function()
        self.init_done = True






####
    def _use_flat_warmup(self):
        return (
            hasattr(self.cfg.terrain, "flat_warmup_iters")
            and self.ppo_it < self.cfg.terrain.flat_warmup_iters
        )

    def _max_allowed_terrain_level(self):
        warmup = getattr(self.cfg.terrain, "flat_warmup_iters", 1000)
        ramp = getattr(self.cfg.terrain, "terrain_ramp_iters", 1000)

        if self.ppo_it < warmup:
            return 0

        progress = min((self.ppo_it - warmup) / ramp, 1.0)
        return int(progress * self.max_terrain_level)


    def _wrap_to_pi(self, x):
        return torch.remainder(x + math.pi, 2 * math.pi) - math.pi

    def _quat_from_axis_angle(self, axis, angle):
        """
        axis: (N, 3)
        angle: (N,)
        return quat: (N, 4) in IsaacGym format [x, y, z, w]
        """
        axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-8)
        half = angle * 0.5
        sin_half = torch.sin(half).unsqueeze(-1)
        cos_half = torch.cos(half).unsqueeze(-1)
        quat_xyz = axis * sin_half
        quat_w = cos_half
        return torch.cat([quat_xyz, quat_w], dim=-1)

    def _tf_combine(self, p_a, q_a, p_b, q_b):
        """
        combine transform T_ab = T_a * T_b

        p_a: (N, 3), q_a: (N, 4)
        p_b: (N, 3), q_b: (N, 4)
        """
        p_ab = p_a + quat_apply(q_a, p_b)
        q_ab = quat_mul(q_a, q_b)
        return p_ab, q_ab

    def _fk_arm_chain(self, joint_pos_6):
        """
        链式 FK：算末端位姿的同时记录每个关节在 mount 系下的原点与轴向，供几何雅可比使用。

        joint_pos_6: (N, 6) = zarx_j1 ~ zarx_j6
        return:
            ee_pos:     (N, 3)
            ee_quat:    (N, 4)
            joint_pos:  (N, 6, 3)  关节 j 原点在 mount 系下的位置
            joint_axis: (N, 6, 3)  关节 j 轴向在 mount 系下的单位向量
        """
        n = joint_pos_6.shape[0]

        p = torch.zeros(n, 3, device=self.device, dtype=torch.float)
        q = torch.zeros(n, 4, device=self.device, dtype=torch.float)
        q[:, 3] = 1.0  # identity

        joint_pos = []
        joint_axis = []

        for j in range(6):
            xyz_j = self.arm_joint_origin_xyz[j].unsqueeze(0).repeat(n, 1)

            rpy_j = self.arm_joint_origin_rpy[j]
            q_origin = quat_from_euler_xyz(
                torch.full((n,), rpy_j[0].item(), device=self.device),
                torch.full((n,), rpy_j[1].item(), device=self.device),
                torch.full((n,), rpy_j[2].item(), device=self.device),
            )

            axis_j = self.arm_joint_axes[j].unsqueeze(0).repeat(n, 1)

            # URDF: T = Trans(origin.xyz) * Rot(origin.rpy) * Rot(axis, q)
            # 关节 j 的坐标系 = 平移 origin.xyz 再转 origin.rpy（尚未施加关节转角）
            p_j = p + quat_apply(q, xyz_j)
            q_j = quat_mul(q, q_origin)

            joint_pos.append(p_j)
            joint_axis.append(quat_apply(q_j, axis_j))

            # 绕自身轴旋转既不移动原点也不改变轴向，故 p 不变、q 右乘关节转角
            q = quat_mul(q_j, self._quat_from_axis_angle(axis_j, joint_pos_6[:, j]))
            p = p_j

        return p, q, torch.stack(joint_pos, dim=1), torch.stack(joint_axis, dim=1)

    def _fk_arm_mount_to_ee(self, joint_pos_6):
        """
        从 arm mount(base_link) 到 zarx_body6 原点的 FK

        joint_pos_6: (N, 6) = zarx_j1 ~ zarx_j6
        return:
            ee_pos_mount:  (N, 3)
            ee_quat_mount: (N, 4)
        """
        ee_pos, ee_quat, _, _ = self._fk_arm_chain(joint_pos_6)
        return ee_pos, ee_quat

    def _arm_jacobian(self, ee_pos, joint_pos, joint_axis):
        """
        几何雅可比（mount 系）。转动关节 j：线速度列 z_j × (p_ee − p_j)，角速度列 z_j。

        return: (N, 6, 6)，行 0:3 线速度 / 3:6 角速度，列对应 6 个受控关节
        """
        lever = ee_pos.unsqueeze(1) - joint_pos              # (N, 6, 3)
        jv = torch.cross(joint_axis, lever, dim=-1)          # (N, 6, 3)
        return torch.cat([jv.transpose(1, 2), joint_axis.transpose(1, 2)], dim=1)

    def _command_pose_in_mount_frame(self):
        """把末端位姿命令（定义在 yaw 对齐 base 系）转换到 mount 系，供逆解使用。
        mount 与 base 只差一个平移、姿态相同，所以只需补回 base 的 roll/pitch。"""
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        q_yaw = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        q_b_ya = quat_mul(quat_conjugate(self.base_quat), q_yaw)   # yaw 对齐系 -> base(=mount) 系

        p_target = quat_apply(q_b_ya, self._commands_lpy_to_local_xyz())
        q_target = quat_mul(q_b_ya, self.target_ee_quat)
        return p_target, q_target

    def _solve_arm_ik(self, p_target, q_target, q_init):
        """阻尼最小二乘迭代逆解 Δq = Jᵀ(JJᵀ + λ²I)⁻¹ e，含步长缩放、步长范数限幅与关节限位。"""
        c = self.cfg.arm.ik_control
        lower = self.dof_pos_limits[self.arm_control_dof_slice, 0].unsqueeze(0)
        upper = self.dof_pos_limits[self.arm_control_dof_slice, 1].unsqueeze(0)
        eye = torch.eye(6, device=self.device).unsqueeze(0)
        lam2 = float(c.damping) ** 2

        q = torch.clamp(q_init.clone(), lower, upper)
        for _ in range(int(c.iters)):
            ee_pos, ee_quat, jpos, jaxis = self._fk_arm_chain(q)

            e_pos = p_target - ee_pos
            e_rot = self._quat_log(quat_mul(q_target, quat_conjugate(ee_quat))) * float(c.ori_weight)
            e = torch.cat([e_pos, e_rot], dim=-1).unsqueeze(-1)          # (N, 6, 1)

            J = self._arm_jacobian(ee_pos, jpos, jaxis)                  # (N, 6, 6)
            lhs = torch.bmm(J, J.transpose(1, 2)) + lam2 * eye
            dq = torch.bmm(J.transpose(1, 2), torch.linalg.solve(lhs, e)).squeeze(-1)

            dq = dq * float(c.step_scale)
            nrm = dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dq = dq * torch.clamp(float(c.max_step_norm) / nrm, max=1.0)
            q = torch.clamp(q + dq, lower, upper)
        return q

    def _update_arm_ik(self):
        """每个控制步刷新一次名义逆解。热启动自上一步的解，因此少量迭代即可收敛。"""
        if not self.use_ik_control:
            self.arm_target_vel.zero_()
            return
        p_target, q_target = self._command_pose_in_mount_frame()
        previous = self.arm_q_ik.clone()
        self.arm_q_ik[:] = self._solve_arm_ik(
            p_target, q_target, self.arm_q_ik
        )
        self.arm_target_vel[:] = (self.arm_q_ik - previous) / self.dt

    def _ik_obs_block(self):
        """第四章观测追加块：名义逆解相对当前构型的偏差，告诉策略“模型认为该往哪走”。"""
        return (self.arm_q_ik - self.dof_pos[:, self.arm_control_dof_slice]) * self.obs_scales.dof_pos

    def _compute_payload_state(self):
        """第二章：机械臂对底座的等效负载状态（base 系，已归一化）。

        臂构型变化让等效质心移动、产生重力力矩，加减速产生惯性反作用——这三者就是
        负载真正作用在底座上的物理量。作为黑盒估计的监督真值（仅训练期需要），
        或白盒前馈时直接进观测。

        return: (N, 9) = [p_com(3), tau_g(3), v_com(3)]
        """
        rb = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)

        # 基座与臂连杆一律取自同一份刚体快照。reset 当帧该快照尚未刷新，但内部相对
        # 几何仍自洽；若混用已刷新的 root_states，会算出米级的伪质心。
        base_p, base_q, base_v = rb[:, 0, 0:3], rb[:, 0, 3:7], rb[:, 0, 7:10]

        p_w = rb[:, self.arm_body_indices, 0:3]                      # (N, K, 3)
        v_w = rb[:, self.arm_body_indices, 7:10]
        m = self.arm_body_masses.view(1, -1, 1)                      # (1, K, 1)
        total_m = self.arm_body_mass_sum

        com_w = (p_w * m).sum(dim=1) / total_m
        vcom_w = (v_w * m).sum(dim=1) / total_m

        # 世界系 -> base 系（相对基座，去掉基座自身的位姿与速度）
        p_com = quat_rotate_inverse(base_q, com_w - base_p)
        v_com = quat_rotate_inverse(base_q, vcom_w - base_v)

        # 重力力矩 tau = p_com × (M·g)，重力方向同样投到这一帧的基座系
        g_base = quat_rotate_inverse(base_q, self.gravity_vec)
        tau_g = torch.cross(p_com, total_m * 9.81 * g_base, dim=-1)

        c = self.cfg.arm.payload
        state = torch.cat([
            p_com / c.com_scale,
            tau_g / c.torque_scale,
            v_com / c.com_vel_scale,
        ], dim=-1)
        return torch.clamp(state, -3.0, 3.0)

    def _get_arm_mount_world_pose(self, env_ids):
        """
        arm mount(base_link) 在世界系下的 pose
        """
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        n = env_ids.numel()

        root_pos = self.root_states[env_ids, 0:3]
        root_quat = self.base_quat[env_ids]

        mount_xyz = self.arm_mount_xyz.unsqueeze(0).repeat(n, 1)

        # 这里 mount_rpy = 0，所以 mount_quat_world = root_quat
        mount_pos_world = root_pos + quat_apply(root_quat, mount_xyz)
        mount_quat_world = root_quat

        return mount_pos_world, mount_quat_world

    def _fk_ee_world_pose(self, env_ids):
        """
        用 FK 计算当前 ee(zarx_body6) 原点在世界系下的 pose
        """
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        joint_pos_6 = self.dof_pos[env_ids, self.arm_control_dof_slice]  # (N, 6)

        mount_pos_world, mount_quat_world = self._get_arm_mount_world_pose(env_ids)
        ee_pos_mount, ee_quat_mount = self._fk_arm_mount_to_ee(joint_pos_6)

        ee_pos_world = mount_pos_world + quat_apply(mount_quat_world, ee_pos_mount)
        ee_quat_world = quat_mul(mount_quat_world, ee_quat_mount)

        return ee_pos_world, ee_quat_world


    def _compute_arm_target_from_actions(self, arm_actions):
        """
        arm_actions: (N, 6)
        第三章 "rl"          ：动作是相对当前构型的增量  q = q_current + Δq·action_scale
        第四章 "ik_residual" ：动作是相对解析逆解的残差  q = q_ik      + Δq·residual_scale
        返回：完整 8 维 arm_target_q，后 2 个夹爪固定默认值
        """
        if self.use_ik_control:
            c = self.cfg.arm.ik_control
            # 名义解由运动学模型给出，策略只学模型未建的那部分（基座运动、PD 滞后、模型误差）
            base_q = self.arm_q_ik
            delta_q = torch.clamp(
                arm_actions * float(c.residual_scale),
                -float(c.residual_clip), float(c.residual_clip)
            )
        else:
            base_q = self.dof_pos[:, self.arm_control_dof_slice]   # (N, 6)
            delta_q = arm_actions * self.cfg.control.action_scale

        arm_target_ctrl = base_q + delta_q

        # joint limit clamp
        lower = self.dof_pos_limits[self.arm_control_dof_slice, 0].unsqueeze(0)
        upper = self.dof_pos_limits[self.arm_control_dof_slice, 1].unsqueeze(0)
        arm_target_ctrl = torch.clamp(arm_target_ctrl, lower, upper)

        # 拼回完整 8 维 arm target
        arm_target_q = self.default_dof_pos[:, self.arm_dof_slice].repeat(self.num_envs, 1)

        # 前 6 个由 policy 控
        arm_target_q[:, :6] = arm_target_ctrl

        # 后 2 个夹爪固定在默认位姿
        default_gripper = self.default_dof_pos[:, self.gripper_dof_slice].repeat(self.num_envs, 1)
        arm_target_q[:, 6:] = default_gripper

        return arm_target_q

    def quat_to_angle(self, quat):
        quat = quat.to(self.device)

        y_vector = to_torch([0., 1., 0.], device=self.device).repeat((quat.shape[0], 1))
        z_vector = to_torch([0., 0., 1.], device=self.device).repeat((quat.shape[0], 1))
        x_vector = to_torch([1., 0., 0.], device=self.device).repeat((quat.shape[0], 1))

        roll_vec = quat_apply(quat, y_vector)
        alpha = torch.atan2(roll_vec[:, 2], roll_vec[:, 1])

        pitch_vec = quat_apply(quat, z_vector)
        beta = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2])

        yaw_vec = quat_apply(quat, x_vector)
        gamma = torch.atan2(yaw_vec[:, 1], yaw_vec[:, 0])

        return torch.stack([alpha, beta, gamma], dim=-1)

    def _quat_geodesic(self, q1, q2):
        """两四元数的测地距离（相对旋转角，rad），平滑无奇异，∈[0,π]。"""
        dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
        return 2.0 * torch.acos(dot)

    def _quat_slerp(self, q0, q1, t):
        """球面线性插值。q0,q1:(N,4) [x,y,z,w]；t:(N,1)。"""
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)      # 取最短弧
        dot = dot.abs().clamp(max=1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        near = sin_theta < 1e-6
        w0 = torch.sin((1.0 - t) * theta) / torch.clamp(sin_theta, min=1e-6)
        w1 = torch.sin(t * theta) / torch.clamp(sin_theta, min=1e-6)
        res = w0 * q0 + w1 * q1
        lerp = (1.0 - t) * q0 + t * q1          # 小角度退化为线性
        res = torch.where(near, lerp, res)
        return res / torch.clamp(torch.norm(res, dim=-1, keepdim=True), min=1e-8)

    def _ee_quat_in_yaw_base(self, env_ids):
        """当前末端姿态在 yaw 对齐 base 系下的四元数（与命令/目标同系）。"""
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        q_base_yaw = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        return quat_mul(quat_conjugate(q_base_yaw), self.ee_quat[env_ids])

    def get_alpha_beta_gamma_in_base_coord(self, env_ids):
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        base_quats = quat_from_euler_xyz(
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            yaw
        )

        ee_in_base_quats = quat_mul(
            quat_conjugate(base_quats),
            self.ee_quat[env_ids]
        )

        abg = self.quat_to_angle(ee_in_base_quats)
        return abg
    
    def get_lpy_in_base_coord(self, env_ids):
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._refresh_ee_state()

        mount_pos_world, _ = self._get_arm_mount_world_pose(env_ids)
        ee_world = self.ee_pos[env_ids]

        d_world = ee_world - mount_pos_world

        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        # 只去掉 base yaw，保持你原来命令语义
        x = torch.cos(yaw) * d_world[:, 0] + torch.sin(yaw) * d_world[:, 1]
        y = -torch.sin(yaw) * d_world[:, 0] + torch.cos(yaw) * d_world[:, 1]
        z = d_world[:, 2]

        l = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        p = torch.atan2(z, torch.sqrt(x**2 + y**2 + 1e-8))
        y_aw = torch.atan2(y, x)

        return torch.stack([l, p, y_aw], dim=-1)
    
    def _get_arm_error_obs(self):
        env_ids = torch.arange(self.num_envs, device=self.device)

        # 当前末端状态（建议这里已经是你统一后的 FK 口径）
        cur_lpy = self.get_lpy_in_base_coord(env_ids)                 # (N, 3)
        cur_abg = self.get_alpha_beta_gamma_in_base_coord(env_ids)    # (N, 3)

        # 目标 - 当前：更适合作为“控制误差”输入
        lpy_error = (self.commands_arm_obs[:, 0:3] - cur_lpy) / self.commands_arm_lpy_range
        abg_error = self._wrap_to_pi(self.target_abg - cur_abg) / self.commands_arm_rpy_range

        arm_error_obs = torch.cat((lpy_error, abg_error), dim=-1)     # (N, 6)

        # 防止偶发极值把观测打爆
        arm_error_obs = torch.clamp(arm_error_obs, -3.0, 3.0)
        return arm_error_obs

    def _invalidate_ee_cache(self):
        """dof_pos / 基座位姿变化后调用，使下一次 _refresh_ee_state 重新做 FK。"""
        self._ee_state_fresh = False

    def _refresh_ee_state(self):
        # 末端位姿只取决于 dof_pos 与基座位姿，二者在一个控制步内只变化一次。
        # 这些访问器被 reward / obs / 误差统计反复调用（实测 13 次/步），
        # 缓存后只在状态真正变化时重算。
        if getattr(self, "_ee_state_fresh", False):
            return
        self._ee_state_fresh = True
        self._refresh_ee_state_impl()

    def _refresh_ee_state_impl(self):
        # simulator rigid body pose（仅用于debug）
        rb = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        self.ee_pos_sim[:] = rb[:, self.ee_body_index, 0:3]
        self.ee_quat_sim[:] = rb[:, self.ee_body_index, 3:7]

        # FK pose（训练 / reward / obs / vis 统一使用这个）
        env_ids = torch.arange(self.num_envs, device=self.device)
        ee_pos_fk, ee_quat_fk = self._fk_ee_world_pose(env_ids)

        self.ee_pos[:] = ee_pos_fk
        self.ee_quat[:] = ee_quat_fk



    def _lpy_to_world_xyz(self, env_ids):
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        l = self.commands_arm[env_ids, 0]
        p = self.commands_arm[env_ids, 1]
        y = self.commands_arm[env_ids, 2]

        # yaw-aligned spherical -> local xyz
        x_local = l * torch.cos(p) * torch.cos(y)
        y_local = l * torch.cos(p) * torch.sin(y)
        z_local = l * torch.sin(p)

        mount_pos_world, _ = self._get_arm_mount_world_pose(env_ids)

        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        base_yaw = torch.atan2(forward[:, 1], forward[:, 0])

        x_world = x_local * torch.cos(base_yaw) - y_local * torch.sin(base_yaw) + mount_pos_world[:, 0]
        y_world = x_local * torch.sin(base_yaw) + y_local * torch.cos(base_yaw) + mount_pos_world[:, 1]
        z_world = z_local + mount_pos_world[:, 2]

        return x_world, y_world, z_world
    


    def _quat_log(self, q):
        """
        q: (N, 4), IsaacGym format [x, y, z, w]
        return: (N, 3), so(3) rotation vector
        """
        q = q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)

        # use shortest rotation
        sign = torch.where(q[:, 3:4] < 0.0, -1.0, 1.0)
        q = q * sign

        xyz = q[:, 0:3]
        w = q[:, 3].clamp(-1.0, 1.0)

        sin_half = torch.norm(xyz, dim=-1)
        angle = 2.0 * torch.atan2(sin_half, w)

        axis = xyz / torch.clamp(sin_half.unsqueeze(-1), min=1e-8)
        rotvec = axis * angle.unsqueeze(-1)

        # near identity, avoid numerical noise
        rotvec = torch.where(
            sin_half.unsqueeze(-1) < 1e-6,
            2.0 * xyz,
            rotvec
        )
        return rotvec


    def get_xyz_in_yaw_base_coord(self, env_ids):
        """
        当前 EE 相对 arm mount 的位置，表达在只去掉 base yaw 的局部坐标中。
        和你原来的 get_lpy_in_base_coord 保持相同语义：只去掉 yaw，不去掉 roll/pitch。
        """
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._refresh_ee_state()

        mount_pos_world, _ = self._get_arm_mount_world_pose(env_ids)
        ee_world = self.ee_pos[env_ids]
        d_world = ee_world - mount_pos_world

        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        x = torch.cos(yaw) * d_world[:, 0] + torch.sin(yaw) * d_world[:, 1]
        y = -torch.sin(yaw) * d_world[:, 0] + torch.cos(yaw) * d_world[:, 1]
        z = d_world[:, 2]

        return torch.stack([x, y, z], dim=-1)


    def _commands_lpy_to_local_xyz(self):
        """
        把你的 command [l, p, y] 转成 yaw-aligned local xyz。
        """
        l = self.commands_arm_obs[:, 0]
        p = self.commands_arm_obs[:, 1]
        y = self.commands_arm_obs[:, 2]

        x = l * torch.cos(p) * torch.cos(y)
        yy = l * torch.cos(p) * torch.sin(y)
        z = l * torch.sin(p)

        return torch.stack([x, yy, z], dim=-1)


    def _get_arm_se3_error_obs(self):
        env_ids = torch.arange(self.num_envs, device=self.device)

        # ---------- position error in local xyz ----------
        cur_xyz = self.get_xyz_in_yaw_base_coord(env_ids)
        des_xyz = self._commands_lpy_to_local_xyz()

        pos_error = des_xyz - cur_xyz

        # 建议单独设一个 xyz range；先没有的话可以复用 lpy range 的尺度
        pos_error = pos_error / self.commands_arm_xyz_error_scale

        # ---------- orientation error as SO(3) log ----------
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        base_yaw = torch.atan2(forward[:, 1], forward[:, 0])

        q_base_yaw = quat_from_euler_xyz(
            torch.zeros_like(base_yaw),
            torch.zeros_like(base_yaw),
            base_yaw
        )

        # current EE orientation in yaw-aligned base frame
        q_cur_local = quat_mul(
            quat_conjugate(q_base_yaw),
            self.ee_quat[env_ids]
        )

        # desired EE orientation：直接用真实目标四元数（此前用 quat_from_euler_xyz(abg)
        # 把 abg 当欧拉角是错的，导致策略观测到的姿态误差也被污染）
        q_des_local = self.target_ee_quat

        # q_err maps current -> desired
        q_err = quat_mul(q_des_local, quat_conjugate(q_cur_local))
        ori_error = self._quat_log(q_err)

        ori_error = ori_error / torch.clamp(self.commands_arm_rpy_range, min=1e-6)

        arm_error_obs = torch.cat([pos_error, ori_error], dim=-1)
        arm_error_obs = torch.clamp(arm_error_obs, -3.0, 3.0)

        return arm_error_obs



####



    def _load_ik_lib(self):
        """命令来源初始化。command_source:
            'joint_space' —— 关节空间正向采样（默认，覆盖整个可达流形、姿态多样）
            'ik_lib'      —— 从 IK 可达性库取配置
            'random'      —— 旧的独立随机采样（位姿大多不可达，仅作对照）
        """
        self.command_source = getattr(self.cfg.arm.commands, "command_source", "random")
        # 关节空间采样用的受控关节 limit
        self.arm_q_lower = self.dof_pos_limits[self.arm_control_dof_slice, 0]
        self.arm_q_upper = self.dof_pos_limits[self.arm_control_dof_slice, 1]
        ls = getattr(self.cfg.arm.commands, "joint_sample_l_range", [0.3, 0.65])
        self.joint_sample_l_min, self.joint_sample_l_max = float(ls[0]), float(ls[1])

        # ik_lib 增广参数
        self.ik_interp_alpha_max = float(getattr(self.cfg.arm.commands, "ik_interp_alpha_max", 0.5))
        # 按关节分层扰动 std：腕部关节(j5,j6,forearm roll)主要改姿态、几乎不动位置，
        # 给大扰动以补齐"给定位置下的姿态多样性"；近端关节改位置，给小扰动
        ps = getattr(self.cfg.arm.commands, "ik_perturb_std", 0.1)
        if isinstance(ps, (list, tuple)):
            self.ik_perturb_std = torch.tensor(ps, device=self.device, dtype=torch.float).reshape(1, -1)
        else:
            self.ik_perturb_std = float(ps)

        # 局部轨迹跟踪：重采新目标时，命令从当前位姿（起点）平滑过渡到新目标（终点），
        # 而非瞬间跳变，给末端一条可跟踪的局部参考轨迹
        traj_cfg = getattr(self.cfg.arm.commands, "trajectory", None)
        self.use_arm_traj = bool(getattr(traj_cfg, "enabled", False)) if traj_cfg is not None else False
        traj_dur = float(getattr(traj_cfg, "duration_s", 1.0)) if traj_cfg is not None else 1.0
        self.arm_traj_T = max(int(round(traj_dur / self.dt)), 1)
        # 轨迹起点/终点：位置用 lpy，姿态用四元数（yaw 对齐 base 系）
        self.arm_traj_start_lpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.arm_traj_end_lpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.arm_traj_start_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.arm_traj_start_quat[:, 3] = 1.0
        self.arm_traj_end_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.arm_traj_end_quat[:, 3] = 1.0
        self.arm_traj_t = torch.full((self.num_envs,), float(self.arm_traj_T), device=self.device)
        # 当前参考姿态（yaw 对齐 base 系四元数），reward/误差用它做测地距离
        self.target_ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_ee_quat[:, 3] = 1.0

        # 躯干/髋部包围盒拒绝：末端落在该盒内（base 系）的命令视为自碰撞，拒绝
        self.body_reject = bool(getattr(self.cfg.arm.commands, "body_reject", True))
        bx = getattr(self.cfg.arm.commands, "body_box_x", [-0.25, 0.30])
        by = getattr(self.cfg.arm.commands, "body_box_y", [-0.12, 0.12])
        bz = getattr(self.cfg.arm.commands, "body_box_z", [-0.15, 0.12])
        self.body_box = torch.tensor(
            [bx, by, bz], device=self.device, dtype=torch.float
        )  # (3,2): [[xmin,xmax],[ymin,ymax],[zmin,zmax]]

        self.ik_lib_q = None
        if self.command_source in ("ik_lib", "ik_lib_aug"):
            default_path = os.path.join(
                LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", "go2_arm", "ee_lib_link6.npz"
            )
            path = getattr(self.cfg.arm.commands, "ik_lib_path", default_path)
            path = path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            if not os.path.isfile(path):
                print(f"[CmdSrc] IK 库文件不存在，回退到关节空间采样: {path}")
                self.command_source = "joint_space"
            else:
                data = np.load(path, allow_pickle=True)
                q = np.asarray(data["q"], dtype=np.float32)
                self.ik_lib_q = torch.from_numpy(q[:, :6]).to(self.device).float()
                self.ik_lib_size = self.ik_lib_q.shape[0]
                if self.command_source == "ik_lib_aug":
                    print(f"[CmdSrc] ik_lib_aug: {self.ik_lib_size} 条可达配置 + 库内插值"
                          f"(alpha_max={self.ik_interp_alpha_max}) + 局部扰动(std={self.ik_perturb_std})")
                else:
                    print(f"[CmdSrc] ik_lib: 从 {os.path.basename(path)} 加载 {self.ik_lib_size} 条可达配置")
        if self.command_source == "joint_space":
            print(f"[CmdSrc] joint_space: 关节空间正向采样，l∈[{self.joint_sample_l_min},{self.joint_sample_l_max}]")

    def _sample_q_joint_space(self, n, max_tries=8):
        """在受控关节 limit 内均匀采样 q，并按 FK 末端距离 l 做拒绝采样，
        挡掉叠在身体里/距离过近过远的怪构型。"""
        lower = self.arm_q_lower.unsqueeze(0)
        upper = self.arm_q_upper.unsqueeze(0)
        q = lower + (upper - lower) * torch.rand(n, 6, device=self.device)
        for _ in range(max_tries):
            ee_pos, _ = self._fk_arm_mount_to_ee(q)
            l = torch.norm(ee_pos, dim=1)
            bad = (l < self.joint_sample_l_min) | (l > self.joint_sample_l_max) | self._ee_in_body(ee_pos)
            if not bad.any():
                break
            m = int(bad.sum())
            q[bad] = lower + (upper - lower) * torch.rand(m, 6, device=self.device)
        return q

    def _ee_in_body(self, ee_pos_mount):
        """末端点（mount 系）换算到 base 系后，是否落在躯干/髋部包围盒内。
        盒内 => 物理自碰撞、不可达，需拒绝。"""
        if not self.body_reject:
            return torch.zeros(ee_pos_mount.shape[0], dtype=torch.bool, device=self.device)
        # mount 系 -> base 系（mount 相对 base 仅有平移 arm_mount_xyz）
        ee_base = ee_pos_mount + self.arm_mount_xyz.unsqueeze(0)
        inside = (
            (ee_base[:, 0] > self.body_box[0, 0]) & (ee_base[:, 0] < self.body_box[0, 1]) &
            (ee_base[:, 1] > self.body_box[1, 0]) & (ee_base[:, 1] < self.body_box[1, 1]) &
            (ee_base[:, 2] > self.body_box[2, 0]) & (ee_base[:, 2] < self.body_box[2, 1])
        )
        return inside

    def _sample_q_ik_aug(self, n, max_tries=8):
        """以 IK 库为骨架做增广：两条库配置插值 + 局部高斯扰动，
        把离散的可达点扩成连续区域，增加覆盖但不离开自然可达流形。
        插值是两个 box-limit 内配置的凸组合，仍在 limit 内；扰动后再 clamp。"""
        lower = self.arm_q_lower.unsqueeze(0)
        upper = self.arm_q_upper.unsqueeze(0)

        def draw(m):
            a = torch.randint(0, self.ik_lib_size, (m,), device=self.device)
            b = torch.randint(0, self.ik_lib_size, (m,), device=self.device)
            alpha = torch.rand(m, 1, device=self.device) * self.ik_interp_alpha_max
            q = (1.0 - alpha) * self.ik_lib_q[a] + alpha * self.ik_lib_q[b]   # 库内插值
            q = q + self.ik_perturb_std * torch.randn(m, 6, device=self.device)  # 局部扰动
            return torch.clamp(q, lower, upper)

        q = draw(n)
        for _ in range(max_tries):
            ee_pos, _ = self._fk_arm_mount_to_ee(q)
            l = torch.norm(ee_pos, dim=1)
            bad = (l < self.joint_sample_l_min) | (l > self.joint_sample_l_max) | self._ee_in_body(ee_pos)
            if not bad.any():
                break
            q[bad] = draw(int(bad.sum()))
        return q

    def _write_arm_command(self, env_ids, lpy, quat):
        """写命令缓冲：位置 lpy + 姿态四元数 quat（yaw 对齐 base 系）。
        target_ee_quat 供 reward/误差做测地距离；abg 仅作为 obs 描述符/可视化。"""
        abg = self.quat_to_angle(quat)
        self.commands_arm[env_ids, 0:3] = lpy
        self.commands_arm_obs[env_ids, 0:3] = lpy
        self.commands_arm_obs[env_ids, 3:6] = abg
        self.target_ee_quat[env_ids] = quat
        self.target_abg[env_ids] = abg
        self.target_rpy[env_ids] = abg
        self.visual_rpy[env_ids] = abg
        self.obj_quats[env_ids] = quat

    def _compute_traj_command(self, env_ids):
        """按进度 arm_traj_t 平滑插值（smoothstep）：位置 lpy 线性(yaw 最短角)，
        姿态四元数 slerp，写入命令缓冲。"""
        s = torch.clamp(self.arm_traj_t[env_ids] / self.arm_traj_T, 0.0, 1.0)
        s = (s * s * (3.0 - 2.0 * s)).unsqueeze(-1)          # smoothstep 缓入缓出
        start_lpy = self.arm_traj_start_lpy[env_ids]
        end_lpy = self.arm_traj_end_lpy[env_ids]
        d = end_lpy - start_lpy
        d[:, 2] = self._wrap_to_pi(d[:, 2])                  # yaw 最短角
        lpy = start_lpy + s * d
        lpy[:, 2] = self._wrap_to_pi(lpy[:, 2])
        quat = self._quat_slerp(self.arm_traj_start_quat[env_ids],
                                self.arm_traj_end_quat[env_ids], s)
        self._write_arm_command(env_ids, lpy, quat)

    def _advance_arm_trajectory(self):
        """每步推进所有环境的轨迹进度并刷新命令（轨迹关闭时为静态目标，进度恒为 T）。"""
        torch.clamp(
            self.arm_traj_t + 1.0,
            max=float(self.arm_traj_T),
            out=self.arm_traj_t,
        )
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._compute_traj_command(all_ids)

    def _set_arm_commands_from_q(self, env_ids, q6, use_traj=True):
        """用一条可达关节配置 q 的 FK 末端位姿作为命令终点（位置 lpy + 姿态四元数，
        联合可达）。use_traj=True 时从当前命令平滑过渡到该终点；否则直接跳变。"""
        ee_pos, ee_quat = self._fk_arm_mount_to_ee(q6)   # mount(=base) 系下末端位姿

        x, y, z = ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2]
        l = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-8)
        p = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2 + 1e-8))
        yaw_pos = torch.atan2(y, x)
        end_lpy = torch.stack([l, p, yaw_pos], dim=-1)
        end_quat = ee_quat                               # yaw 对齐 base 系（水平机身近似）

        self.arm_traj_end_lpy[env_ids] = end_lpy
        self.arm_traj_end_quat[env_ids] = end_quat

        if self.use_arm_traj and use_traj:
            # 起点 = 当前参考（settle/定时重采时臂已大致到达，参考连续）
            self.arm_traj_start_lpy[env_ids] = self.commands_arm_obs[env_ids, 0:3].clone()
            self.arm_traj_start_quat[env_ids] = self.target_ee_quat[env_ids].clone()
            self.arm_traj_t[env_ids] = 0.0
            self._compute_traj_command(env_ids)          # 立即写入 s=0 的起点命令
        else:
            self.arm_traj_start_lpy[env_ids] = end_lpy
            self.arm_traj_start_quat[env_ids] = end_quat
            self.arm_traj_t[env_ids] = float(self.arm_traj_T)
            self._write_arm_command(env_ids, end_lpy, end_quat)   # 直接跳到终点

    def _resample_arm_commands(self, env_ids, use_traj=True):
        if len(env_ids) == 0:
            return
        if not self._arm_motion_enabled():
            self._set_default_arm_command(env_ids)
            return

        curriculum = getattr(self, "arm_trajectory_curriculum", None)
        if curriculum is not None and curriculum.enabled:
            q6 = curriculum.sample_q(self, env_ids)
            self._set_arm_commands_from_q(
                env_ids, q6, use_traj=use_traj
            )
            curriculum.begin_trajectories(env_ids)
            return

        # 可达命令采样：位置和姿态由同一条真实关节配置 q 的 FK 导出，联合可达
        src = getattr(self, "command_source", "random")
        if src == "joint_space":
            q6 = self._sample_q_joint_space(len(env_ids))
            self._set_arm_commands_from_q(env_ids, q6, use_traj=use_traj)
            return
        if src == "ik_lib" and self.ik_lib_q is not None:
            idx = torch.randint(0, self.ik_lib_size, (len(env_ids),), device=self.device)
            self._set_arm_commands_from_q(env_ids, self.ik_lib_q[idx], use_traj=use_traj)
            return
        if src == "ik_lib_aug" and self.ik_lib_q is not None:
            self._set_arm_commands_from_q(env_ids, self._sample_q_ik_aug(len(env_ids)), use_traj=use_traj)
            return

        # position command in spherical coordinates
        self.commands_arm[env_ids, 0] = torch_rand_float(
            self.cfg.arm.commands.l[0], self.cfg.arm.commands.l[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.commands_arm[env_ids, 1] = torch_rand_float(
            self.cfg.arm.commands.p[0], self.cfg.arm.commands.p[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.commands_arm[env_ids, 2] = torch_rand_float(
            self.cfg.arm.commands.y[0], self.cfg.arm.commands.y[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # orientation command in rpy
        roll = torch_rand_float(
            self.cfg.arm.commands.roll_ee[0], self.cfg.arm.commands.roll_ee[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)

        pitch = torch_rand_float(
            self.cfg.arm.commands.pitch_ee[0], self.cfg.arm.commands.pitch_ee[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)

        yaw = torch_rand_float(
            self.cfg.arm.commands.yaw_ee[0], self.cfg.arm.commands.yaw_ee[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.target_rpy[env_ids, 0] = roll
        self.target_rpy[env_ids, 1] = pitch
        self.target_rpy[env_ids, 2] = yaw
        self.visual_rpy[env_ids] = self.target_rpy[env_ids]

        # 给观测的命令版本
        self.commands_arm_obs[env_ids, 0:3] = self.commands_arm[env_ids, 0:3]
        self.commands_arm_obs[env_ids, 3] = roll
        self.commands_arm_obs[env_ids, 4] = pitch
        self.commands_arm_obs[env_ids, 5] = yaw


        self.target_abg[env_ids, 0] = roll
        self.target_abg[env_ids, 1] = pitch
        self.target_abg[env_ids, 2] = yaw

        # 存一个目标四元数，后面 reward / 可视化都方便
        q = quat_from_euler_xyz(roll, pitch, yaw)
        self.obj_quats[env_ids] = q.reshape(-1, 4)

    def get_dog_observations(self):
        """Return the frozen base policy's independent proprioceptive history."""
        dog_hist = torch.stack(
            [self.dog_obs_history[index]
             for index in range(self.dog_obs_history.maxlen)],
            dim=1,
        )
        return dog_hist.reshape(self.num_envs, -1)

    def configure_base_policy_observation(self, history_len, obs_dim):
        """Configure the base policy history without changing the arm history.

        The terrain-MoE base was trained on 45 proprioceptive values over 20
        frames, while the arm policy intentionally keeps its six-frame layout.
        """
        history_len = int(history_len)
        obs_dim = int(obs_dim)
        if history_len < 1 or obs_dim < 1:
            raise ValueError(
                f"Invalid base policy observation shape: {history_len}x{obs_dim}"
            )
        if obs_dim > self.num_dog_one_step_obs:
            raise ValueError(
                f"Base policy obs_dim={obs_dim} exceeds dog observation "
                f"dimension {self.num_dog_one_step_obs}"
            )
        self.base_policy_history_len = history_len
        self.base_policy_obs_dim = obs_dim
        self.dog_obs_history = deque(maxlen=history_len)
        for _ in range(history_len):
            self.dog_obs_history.append(
                torch.zeros(
                    self.num_envs,
                    obs_dim,
                    dtype=torch.float,
                    device=self.device,
                )
            )

    def get_curriculum_state(self):
        curriculum = getattr(self, "arm_trajectory_curriculum", None)
        return curriculum.state_dict() if curriculum is not None else None

    def load_curriculum_state(self, state):
        curriculum = getattr(self, "arm_trajectory_curriculum", None)
        if curriculum is not None:
            curriculum.load_state_dict(state)

    def _dog_obs_block(self):
        """底座本体观测块。第二章白盒模式下在尾部追加解析负载状态；
        estimated / none 模式下维持 45 维原布局，冻结底座策略的接口不受影响。"""
        dog_pos = (self.dof_pos[:, self.dog_dof_slice] - self.default_dof_pos[:, self.dog_dof_slice]) * self.obs_scales.dof_pos
        dog_vel = self.dof_vel[:, self.dog_dof_slice] * self.obs_scales.dof_vel

        terms = [
            self.commands[:, :3] * self.commands_scale,   # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.projected_gravity,                       # 3
            dog_pos,                                      # 12
            dog_vel,                                      # 12
            self.dog_actions,                             # 12
        ]
        if self.payload_obs_dim > 0:
            terms.append(self._compute_payload_state())   # 9，仅白盒
        return torch.cat(terms, dim=-1)

    def get_dog_one_step_obs(self):
        return self._dog_obs_block()

    def _privileged_current_obs(self, current_obs):
        """特权观测：本体观测 + 基座真实线速度 + 外扰 + 高度图 (+ 第二章负载真值)。
        负载真值仅在训练期存在，用作黑盒估计器的监督目标。"""
        parts = [
            current_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.disturbance[:, 0, :],
        ]
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            parts.append(heights)
        if self.payload_priv_dim > 0:
            parts.append(self._compute_payload_state())
        return torch.cat(parts, dim=-1)



    def set_dog_actions(self, dog_actions):
        self.dog_actions[:] = dog_actions

    def set_dog_residual(self, dog_residual):
        self.dog_residual[:] = dog_residual

    def get_base_residual_obs(self):
        """底座顺应残差网络的观测：当前 dog obs + arm 命令 + arm 末端误差。
        让底座'知道'臂要够哪、当前差多少，从而主动下蹲/前倾/挪重心去配合。"""
        dog = self.get_dog_one_step_obs()              # 45
        arm_err = self._get_arm_se3_error_obs()        # 6 (pos 3 + ori 3)
        return torch.cat((dog, self.commands_arm_obs, arm_err), dim=-1)  # 57

    def draw_env0_desired_ee(self):
        if self.viewer is None:
            return

        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)

        # 期望末端位置（世界系）
        x, y, z = self._lpy_to_world_xyz(env_ids)

        # 绿色球：期望末端位置
        sphere_geom = gymutil.WireframeSphereGeometry(
            0.03, 12, 12, None, color=(0, 1, 0)
        )
        sphere_pose = gymapi.Transform(
            gymapi.Vec3(x.item(), y.item(), z.item()), r=None
        )
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

        # ------- 可选：再画期望姿态坐标轴 -------
        roll = self.target_abg[0, 0]
        pitch = self.target_abg[0, 1]
        yaw_cmd = self.target_abg[0, 2]

        # base 当前 yaw
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        base_yaw = torch.atan2(forward[:, 1], forward[:, 0])

        q_base_yaw = quat_from_euler_xyz(
            torch.zeros_like(base_yaw),
            torch.zeros_like(base_yaw),
            base_yaw
        )
        q_cmd = quat_from_euler_xyz(
            roll.unsqueeze(0),
            pitch.unsqueeze(0),
            yaw_cmd.unsqueeze(0)
        )
        q_world = quat_mul(q_base_yaw, q_cmd)[0]   # (4,)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(
            q_world[0].item(),
            q_world[1].item(),
            q_world[2].item(),
            q_world[3].item()
        )
        axes_geom = gymutil.AxesGeometry(0.12, pose)
        axes_pose = gymapi.Transform(
            gymapi.Vec3(x.item(), y.item(), z.item()), r=None
        )
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], axes_pose)


    def draw_env0_current_ee(self):
        if self.viewer is None:
            return

        self._refresh_ee_state()

        # FK 当前末端
        x = self.ee_pos[0, 0].item()
        y = self.ee_pos[0, 1].item()
        z = self.ee_pos[0, 2].item()

        sphere_geom = gymutil.WireframeSphereGeometry(
            0.025, 12, 12, None, color=(1, 1, 0)
        )
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

        # 可选：再画一个 simulator rigid body 读数，方便检查 FK 是否对齐
        x_sim = self.ee_pos_sim[0, 0].item()
        y_sim = self.ee_pos_sim[0, 1].item()
        z_sim = self.ee_pos_sim[0, 2].item()

        sphere_geom_sim = gymutil.WireframeSphereGeometry(
            0.018, 10, 10, None, color=(0, 1, 1)
        )
        sphere_pose_sim = gymapi.Transform(gymapi.Vec3(x_sim, y_sim, z_sim), r=None)
        gymutil.draw_lines(sphere_geom_sim, self.gym, self.viewer, self.envs[0], sphere_pose_sim)
#############################

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        if self.actions.shape[-1] != self.num_dog_actions:
            raise ValueError(
                "Chapter 4 expects 12 Go2 actions, got "
                f"{self.actions.shape[-1]}"
            )
        self.dog_actions[:] = self.actions

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        self.render()


        
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # if self.keep_arm_fixed:
            #     self._keep_arm_fixed()
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        # 物理已推进，末端位姿缓存失效
        self._invalidate_ee_cache()
        self._refresh_ee_state()

        self._post_physics_step_callback()
        self._compute_arm_error_obs()
        curriculum = getattr(self, "arm_trajectory_curriculum", None)
        if curriculum is not None and curriculum.enabled:
            curriculum.observe_payload(
                self._compute_payload_state(), self.dt
            )
        self._update_arm_settle()
        if curriculum is not None and curriculum.enabled:
            curriculum.update_stages(self.ppo_it)

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)

        # --- 事件驱动重采样：settle 的 env 立即换新目标（reset 的 env 已在 reset_idx 处理）---
        if self._arm_motion_enabled():
            settled_ids = (self.settle_buf & ~self.reset_buf).nonzero(as_tuple=False).flatten()
            if len(settled_ids) > 0:
                if curriculum is not None and curriculum.enabled:
                    stable = (
                        (torch.abs(self.roll[settled_ids]) < 0.5)
                        & (torch.abs(self.pitch[settled_ids]) < 0.5)
                        & (self._get_base_heights(settled_ids) > 0.20)
                    )
                    curriculum.record_result(
                        settled_ids,
                        torch.ones_like(stable),
                        stable,
                    )
                self._resample_arm_commands(settled_ids)
                self.arm_settle_counter[settled_ids] = 0

        # 用 resample/reset 后的最新命令与状态刷新 PBRS potential，保证下一步
        # gamma*phi' - phi 两项口径一致
        self._update_arm_err_norms()
        self.arm_potential[:] = self._arm_potential_fn()
        self.arm_pbrs_skip[:] = False

        # 命令定稿后刷新名义逆解：供本步观测与下一步的动作合成使用
        self._update_arm_ik()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]

        self.last_last_dog_actions[:] = self.last_dog_actions[:]
        self.last_dog_actions[:] = self.dog_actions[:]

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # --- NEW: update history buffers for rewards (only for envs that are NOT reset this step) ---
        not_reset_ids = (~self.reset_buf).nonzero(as_tuple=False).flatten()
        if len(not_reset_ids) > 0:
            self.prev_contact_mask[not_reset_ids] = self.contact_mask[not_reset_ids]
            self.last_base_ang_vel[not_reset_ids] = self.base_ang_vel[not_reset_ids]


        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        # Publish the current-step mask even when no environment resets.
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone()
        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.reverse_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)        
        rpy = quaternion_to_rpy(self.base_quat)
        self.roll, self.pitch, self.y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        # --- NEW: roll / pitch termination (base only) ---
        # 联合 fine-tune 时放宽阈值：允许底座下蹲/前倾去配合机械臂
        jf = getattr(self.cfg.arm, "joint_finetune", None)
        if jf is not None and getattr(jf, "enabled", False):
            roll_th = getattr(jf, "roll_term", 1.2)
            pitch_th = getattr(jf, "pitch_term", 1.2)
        else:
            roll_th = 0.8  # 45 deg in rad
            pitch_th = 0.8 # 45 deg in rad

        bad_roll  = torch.abs(self.roll)  > roll_th
        bad_pitch = torch.abs(self.pitch) > pitch_th

        termination_cfg = self.cfg.termination
        if self.termination_contact_indices.numel() > 0:
            contact_force = torch.norm(
                self.contact_forces[
                    :, self.termination_contact_indices, :
                ],
                p=2,
                dim=-1,
            )
            self.base_contact_buf = torch.any(
                contact_force
                > float(termination_cfg.base_contact_force_threshold),
                dim=1,
            )
        else:
            self.base_contact_buf.zero_()
        self.base_contact_counter = torch.where(
            self.base_contact_buf,
            self.base_contact_counter + 1,
            torch.zeros_like(self.base_contact_counter),
        )
        self.base_contact_termination_buf = (
            self.base_contact_counter
            >= int(termination_cfg.base_contact_persist_steps)
        )
        self.current_base_height = self._get_base_heights()

        hard_fall = (
            bad_roll
            | bad_pitch
            | self.base_contact_termination_buf
        )
        self.roll_termination_buf.copy_(bad_roll)
        self.pitch_termination_buf.copy_(bad_pitch)
        self.fall_termination_buf.copy_(hard_fall)
        # Timeout is a truncation, not a physical terminal state. A fall on
        # the same step takes precedence and must not bootstrap.
        self.time_out_buf = (
            self.episode_length_buf >= self.max_episode_length
        ) & ~hard_fall
        self.reset_buf = hard_fall | self.time_out_buf

        # if(self.ppo_it>self.cfg.arm.curriculum_iters[0]):
        #     self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) < 0.23
        #     self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

    def _arm_warmup_active(self):
        return (
            self.arm_fixed_warmup_iters > 0
            and self.ppo_it < self.arm_fixed_warmup_iters
        )

    def _arm_motion_enabled(self):
        return (
            not self.keep_arm_fixed
            and not self._arm_warmup_active()
        )

    def _set_default_arm_command(self, env_ids):
        if len(env_ids) == 0:
            return
        q6 = self.default_dof_pos[
            :, self.arm_control_dof_slice
        ].expand(len(env_ids), -1)
        self._set_arm_commands_from_q(
            env_ids, q6, use_traj=False
        )

    def _keep_arm_fixed(self):
        arm_start = 12  # 前12是狗，后8是臂
        self.dof_pos[:, arm_start:] = self.default_dof_pos[:, arm_start:]
        self.dof_vel[:, arm_start:] = 0.0

        # 写回仿真
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self._invalidate_ee_cache()


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # Never expose episode statistics from a previous reset step.
        self.extras.pop("episode", None)
        if len(env_ids) == 0:
            return
        curriculum = getattr(self, "arm_trajectory_curriculum", None)
        if (
            curriculum is not None
            and curriculum.enabled
            and self.init_done
            and hasattr(self, "roll")
        ):
            stable = (
                (torch.abs(self.roll[env_ids]) < 0.5)
                & (torch.abs(self.pitch[env_ids]) < 0.5)
                & (self._get_base_heights(env_ids) > 0.20)
            )
            curriculum.record_result(
                env_ids,
                self.settle_buf[env_ids],
                stable,
            )
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # 关节角与基座位姿被改写，末端位姿缓存失效
        self._invalidate_ee_cache()

        # --- NEW: reset history buffers for these envs ---
        self.prev_contact_mask[env_ids] = False

        # set last_base_ang_vel to the (new) reset state's ang vel in body frame to avoid a spike
        base_quat = self.root_states[env_ids, 3:7]
        ang_vel_body = quat_rotate_inverse(base_quat, self.root_states[env_ids, 10:13])
        self.last_base_ang_vel[env_ids] = ang_vel_body


        self._resample_commands(env_ids)
        if self._arm_motion_enabled():
            # reset 时直接给静态目标（臂从默认姿态起步，不走轨迹）
            self._resample_arm_commands(env_ids, use_traj=False)
        else:
            self._set_default_arm_command(env_ids)

        # 逆解热启动重置为默认臂姿，避免沿用上一回合的解
        self.arm_q_ik[env_ids] = self.default_dof_pos[:, self.arm_control_dof_slice]
        self.arm_target_vel[env_ids] = 0.0

        # settle / PBRS buffers
        self.arm_settle_counter[env_ids] = 0
        # reset 发生在 reward 之后、potential 刷新之前；但初始 reset()（不经过
        # post_physics_step）后第一步的 PBRS 口径不一致，统一跳过一步
        self.arm_pbrs_skip[env_ids] = True

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.

        self.dog_actions[env_ids] = 0.
        self.dog_residual[env_ids] = 0.
        self.last_dog_actions[env_ids] = 0.
        self.last_last_dog_actions[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.base_contact_buf[env_ids] = False
        self.base_contact_counter[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_buf[env_ids] = 0.

        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.

        # 分组奖励 episode 均值；单评论家也记录原始三组，便于公平比较。
        for g in self.reward_groups:
            self.extras["episode"]['rewgroup_' + g] = torch.mean(
                self.episode_sums_group[g][env_ids]
                / torch.clip(self.episode_length_buf[env_ids], min=1)
                / self.dt
            )
            self.episode_sums_group[g][env_ids] = 0.

        # arm 跟踪硬指标：episode 平均误差 [m]/[rad]、settle 率 [次/s]、当前课程容差
        ep_len = torch.clip(self.episode_length_buf[env_ids], min=1).float()
        self.extras["episode"]["arm_pos_err_mean"] = torch.mean(self.episode_pos_err_sum[env_ids] / ep_len)
        self.extras["episode"]["arm_ori_err_mean"] = torch.mean(self.episode_ori_err_sum[env_ids] / ep_len)
        self.extras["episode"]["arm_settle_per_s"] = torch.mean(self.episode_settle_count[env_ids] / (ep_len * self.dt))
        self.extras["episode"]["arm_pos_tol"] = self.settle_pos_tol
        self.extras["episode"]["arm_ori_tol"] = self.settle_ori_tol
        self.extras["episode"]["arm_warmup_active"] = float(
            self._arm_warmup_active()
        )
        self.extras["episode"]["base_contact_termination_rate"] = torch.mean(
            self.base_contact_termination_buf[env_ids].float()
        )
        self.extras["episode"]["roll_termination_rate"] = torch.mean(
            self.roll_termination_buf[env_ids].float()
        )
        self.extras["episode"]["pitch_termination_rate"] = torch.mean(
            self.pitch_termination_buf[env_ids].float()
        )
        self.extras["episode"]["fall_termination_rate"] = torch.mean(
            self.fall_termination_buf[env_ids].float()
        )
        self.extras["episode"]["timeout_rate"] = torch.mean(
            self.time_out_buf[env_ids].float()
        )
        self.extras["episode"]["base_height"] = torch.mean(
            self.current_base_height[env_ids]
        )
        self.extras["episode"]["_episode_count"] = int(len(env_ids))
        self.base_contact_termination_buf[env_ids] = False
        self.episode_pos_err_sum[env_ids] = 0.
        self.episode_ori_err_sum[env_ids] = 0.
        self.episode_settle_count[env_ids] = 0.
        if curriculum is not None and curriculum.enabled:
            curriculum_metrics = curriculum.metadata()
            self.extras["episode"]["curriculum_stage"] = (
                curriculum_metrics["stage"][env_ids].float().mean()
            )
            self.extras["episode"]["curriculum_success_rate"] = (
                curriculum_metrics["mean_success_rate"]
            )
            self.extras["episode"]["curriculum_stability_rate"] = (
                curriculum_metrics["mean_stability_rate"]
            )
            self.extras["episode"]["curriculum_load_difficulty"] = (
                curriculum_metrics["mean_load_difficulty"]
            )
            self.extras["episode"]["curriculum_ik_success_rate"] = (
                curriculum_metrics["ik_success_rate"]
            )
            self.extras["episode"]["curriculum_settle_rate"] = (
                curriculum_metrics["mean_success_rate"]
            )
            self.extras["episode"]["curriculum_trajectory_completion_rate"] = (
                curriculum_metrics["mean_success_rate"]
            )
            for stage_index, ratio in enumerate(
                curriculum_metrics["stage_sampling"]
            ):
                self.extras["episode"][
                    f"curriculum_stage_sampling_{stage_index}"
                ] = ratio
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        if hasattr(self, "dog_obs_history"):
            for frame in self.dog_obs_history:
                frame[env_ids] *= 0

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_vec[:] = 0.
        group_rew = torch.zeros(
            self.num_envs,
            len(self.reward_groups),
            device=self.device,
        )
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            g = self.reward_group_idx[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            group_rew[:, g] += rew
            self.episode_sums[name] += rew
            self.episode_sums_group[self.reward_groups[g]] += rew
        if self.cfg.rewards.only_positive_rewards:
            # 只对 task 组截断：stab / energy 本身就是代价项，截断会让它们失效
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            group_rew[:, 0] = torch.clip(group_rew[:, 0], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            # 终止惩罚始终属于 stability 组。
            g_term = self.reward_groups.index("stab")
            group_rew[:, g_term] += rew
            self.episode_sums["termination"] += rew
            self.episode_sums_group[self.reward_groups[g_term]] += rew

        if self.value_dim > 1:
            self.rew_buf_vec[:, :] = group_rew[:, :self.value_dim]
        else:
            weighted = (
                group_rew * self.reward_group_weights.unsqueeze(0)
            ).sum(dim=-1)
            self.rew_buf[:] = weighted
            self.rew_buf_vec[:, 0] = weighted
    
    def compute_observations(self):
        self._refresh_ee_state()

        current_obs = self._dog_obs_block()

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[:current_obs.shape[1]]

        self.dog_obs_history.append(current_obs)
        self.obs_history.append(current_obs)
        obs_history = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_history.reshape(self.num_envs, -1)

        privileged_current_obs = self._privileged_current_obs(current_obs)

        self.privileged_obs_buf = torch.cat((
            privileged_current_obs[:, :self.num_one_step_privileged_obs],
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs],
            self.contact_mask
        ), dim=-1)



    def _compute_arm_error_obs(self):
        env_ids = torch.arange(self.num_envs, device=self.device)

        # 当前末端 pose，建议用你已经验证过的 FK，而不是 rigid body 读数
        ee_lpy = self.get_lpy_in_base_coord(env_ids)
        ee_abg = self.get_alpha_beta_gamma_in_base_coord(env_ids)

        pos_error = self.commands_arm_obs[:, 0:3] - ee_lpy
        ori_error = self._wrap_to_pi(self.target_abg - ee_abg)

        self.arm_d_pos_error[:] = pos_error - self.arm_pos_error
        self.arm_d_ori_error[:] = self._wrap_to_pi(ori_error - self.arm_ori_error)

        self.last_arm_pos_error[:] = self.arm_pos_error
        self.last_arm_ori_error[:] = self.arm_ori_error

        self.arm_pos_error[:] = pos_error
        self.arm_ori_error[:] = ori_error

    # ------------ 事件驱动 settle + PBRS ----------------
    def _update_arm_err_norms(self):
        """欧氏位置误差 [m] + 姿态误差 [rad]，用于 settle 判定和 PBRS potential"""
        env_ids = torch.arange(self.num_envs, device=self.device)

        cur_xyz = self.get_xyz_in_yaw_base_coord(env_ids)
        des_xyz = self._commands_lpy_to_local_xyz()
        self.arm_pos_err_norm[:] = torch.norm(des_xyz - cur_xyz, dim=1)

        # 姿态误差用四元数测地距离（rad），无奇异、忠实反映真实旋转误差
        cur_quat = self._ee_quat_in_yaw_base(env_ids)
        self.arm_ori_err_norm[:] = self._quat_geodesic(cur_quat, self.target_ee_quat)

    def _arm_potential_fn(self):
        return -(
            self.arm_pos_err_norm
            + self.cfg.arm.settle.pbrs_ori_weight * self.arm_ori_err_norm
        )

    def _update_arm_settle(self):
        """每步 reward 之前调用：更新 settle 计数与本步 potential"""
        self._update_arm_err_norms()

        if not self._arm_motion_enabled():
            self.arm_settle_counter.zero_()
            self.settle_buf.zero_()
            self.arm_potential_now[:] = self._arm_potential_fn()
            self.arm_pbrs_skip[:] = True
            return

        in_pos = self.arm_pos_err_norm < self.settle_pos_tol
        in_ori = self.arm_ori_err_norm < self.settle_ori_tol
        if self.use_arm_traj:
            trajectory_finished = self.arm_traj_t >= float(self.arm_traj_T)
        else:
            trajectory_finished = torch.ones_like(in_pos)
        in_tol = in_pos & in_ori & trajectory_finished
        self.arm_settle_counter = torch.where(
            in_tol,
            self.arm_settle_counter + 1,
            torch.zeros_like(self.arm_settle_counter),
        )
        # settle 当步发放 bonus，随后立即重采样并清零计数（一次性事件）
        self.settle_buf = self.arm_settle_counter >= self.cfg.arm.settle.hold_steps

        self.arm_potential_now[:] = self._arm_potential_fn()

        # 轨迹参考仍在移动时跳过 PBRS：PBRS 只对静态目标收敛有效，
        # 否则目标移动会混入虚假的进展/倒退信号
        if self.use_arm_traj:
            self.arm_pbrs_skip |= (self.arm_traj_t < self.arm_traj_T)

        # episode 级日志统计
        self.episode_pos_err_sum += self.arm_pos_err_norm
        self.episode_ori_err_sum += self.arm_ori_err_norm
        self.episode_settle_count += self.settle_buf.float()

        # 容差课程统计：按轴独立累计"在该轴容差内的步数占比"
        self._settle_events_acc += self.settle_buf.sum()
        self._pos_intol_acc += in_pos.sum()
        self._ori_intol_acc += in_ori.sum()
        self._settle_steps_acc += 1
        self._maybe_update_settle_curriculum()

    def _maybe_update_settle_curriculum(self):
        """每 curriculum_interval 个 PPO 迭代评估一次；位置/姿态容差按各自轴的
        在容差内时间占比独立收紧，避免姿态轴拖累位置轴一起停滞。"""
        s = self.cfg.arm.settle
        if not getattr(s, "curriculum", False):
            return
        if self.ppo_it <= 0 or self.ppo_it == self._last_settle_curriculum_it:
            return
        if self.ppo_it % s.curriculum_interval != 0:
            return
        self._last_settle_curriculum_it = self.ppo_it

        denom = max(self.num_envs * self._settle_steps_acc, 1)
        pos_intol_rate = self._pos_intol_acc.item() / denom
        ori_intol_rate = self._ori_intol_acc.item() / denom

        if pos_intol_rate >= s.pos_intol_target:
            self.settle_pos_tol = max(self.settle_pos_tol * s.tol_decay, s.pos_tol_final)
        if ori_intol_rate >= s.ori_intol_target:
            self.settle_ori_tol = max(self.settle_ori_tol * s.tol_decay, s.ori_tol_final)

        self._settle_events_acc.zero_()
        self._pos_intol_acc.zero_()
        self._ori_intol_acc.zero_()
        self._settle_steps_acc = 0

    def get_current_obs(self):
        self._refresh_ee_state()
        current_obs = self._dog_obs_block()

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[:current_obs.shape[1]]

        return self._privileged_current_obs(current_obs)
        
    def compute_termination_observations(self, env_ids):
        self._refresh_ee_state()

        current_obs = self._dog_obs_block()

        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[:current_obs.shape[1]]

        privileged_current_obs = self._privileged_current_obs(current_obs)

        return torch.cat((
            privileged_current_obs[:, :self.num_one_step_privileged_obs],
            self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs],
            self.contact_mask
        ), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        warmup_active = self._arm_warmup_active()
        if (
            not self.keep_arm_fixed
            and self._arm_warmup_was_active
            and not warmup_active
        ):
            all_ids = torch.arange(
                self.num_envs, device=self.device
            )
            self._resample_arm_commands(all_ids)
            self.arm_settle_counter.zero_()
            self.arm_pbrs_skip[:] = True
        self._arm_warmup_was_active = warmup_active

        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()

        # ② 再 resample 新目标
        self._resample_commands(env_ids)
        if self._arm_motion_enabled():
            self._resample_arm_commands(env_ids)
            if len(env_ids) > 0:
                # 定时重采样发生在 reward 之前：本步 PBRS 口径不一致，置零跳过
                self.arm_settle_counter[env_ids] = 0
                self.arm_pbrs_skip[env_ids] = True

        # 每步推进局部轨迹参考（命令从起点平滑移向终点）
        if (
            self._arm_motion_enabled()
            and self.use_arm_traj
            and self.command_source != "random"
        ):
            self._advance_arm_trajectory()

        self.contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)

        # set y commands of high vel envs to zero
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, dog_actions):
        torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        # -------------------------
        # dog torques: PPO base actor output
        # -------------------------
        dog_actions_scaled = dog_actions * self.cfg.control.action_scale
        dog_target = self.default_dof_pos[:, self.dog_dof_slice] + dog_actions_scaled

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques[:, self.dog_dof_slice] = (
                self.p_gains[self.dog_dof_slice].unsqueeze(0) * self.Kp_factors *
                (dog_target - self.dof_pos[:, self.dog_dof_slice])
                - self.d_gains[self.dog_dof_slice].unsqueeze(0) * self.Kd_factors *
                self.dof_vel[:, self.dog_dof_slice]
            )
        elif control_type == "V":
            torques[:, self.dog_dof_slice] = (
                - self.p_gains[self.dog_dof_slice].unsqueeze(0) *
                self.dof_vel[:, self.dog_dof_slice]
                - self.d_gains[self.dog_dof_slice].unsqueeze(0) *
                (self.dof_vel[:, self.dog_dof_slice] - self.last_dof_vel[:, self.dog_dof_slice]) / self.sim_params.dt
            )
        elif control_type == "T":
            torques[:, self.dog_dof_slice] = dog_actions
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        # -------------------------
        # arm torques: fixed trajectory target executed by PD
        # -------------------------
        if not self._arm_motion_enabled():
            arm_target = self.default_dof_pos[:, self.arm_dof_slice].expand(self.num_envs, -1)
            arm_target_vel = torch.zeros_like(arm_target)
        else:
            arm_target = torch.zeros(
                self.num_envs,
                self.num_arm_dofs,
                device=self.device,
            )
            arm_target[:, :6] = self.arm_q_ik
            arm_target[:, 6:] = self.default_dof_pos[
                :, self.gripper_dof_slice
            ].expand(self.num_envs, -1)
            arm_target_vel = torch.zeros_like(arm_target)
            arm_target_vel[:, :6] = self.arm_target_vel

        self.arm_target_q[:] = arm_target

        torques[:, self.arm_dof_slice] = (
            self.p_gains[self.arm_dof_slice].unsqueeze(0) *
            (arm_target - self.dof_pos[:, self.arm_dof_slice])
            + self.d_gains[self.arm_dof_slice].unsqueeze(0) *
            (arm_target_vel - self.dof_vel[:, self.arm_dof_slice])
        )

        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def _reset_dofs(self, env_ids):
        # 全部先回默认
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        # 只随机狗 0:12
        self.dof_pos[env_ids, self.dog_dof_slice] = (
            self.default_dof_pos[:, self.dog_dof_slice] *
            torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dog_actions), device=self.device)
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """
        前 flat_warmup_iters 轮强制使用平地；
        之后恢复原来的 terrain curriculum。
        """
        if not self.init_done:
            return

        # -----------------------------
        # flat warmup stage
        # -----------------------------
        if self._use_flat_warmup():
            flat_level = getattr(self.cfg.terrain, "flat_warmup_level", 0)

            self.terrain_levels[env_ids] = flat_level

            # 保留 terrain_types，让不同 env 仍然分布在不同列；
            # 如果你只想全部放到同一个平地区域，也可以把 terrain_types[env_ids] = 0
            self.env_origins[env_ids] = self.terrain_origins[
                self.terrain_levels[env_ids],
                self.terrain_types[env_ids]
            ]

            return

        # -----------------------------
        # original terrain curriculum
        # -----------------------------
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2],
            dim=1
        )

        move_up = distance > self.terrain.env_length / 2

        move_down = (
            distance < torch.norm(self.commands[env_ids, :2], dim=1)
            * self.max_episode_length_s
            * 0.5
        ) * ~move_up

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )

        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids],
            self.terrain_types[env_ids]
        ]

        max_allowed = self._max_allowed_terrain_level()
        self.terrain_levels[env_ids] = torch.clip(
            self.terrain_levels[env_ids],
            0,
            max_allowed
        )
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(
            int(getattr(cfg.env, "num_dog_one_step_observations", 45)),
            device=self.device,
        )

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        idx = 0

        # commands (3)
        noise_vec[idx:idx+3] = 0.
        idx += 3

        # base_ang_vel (3)
        noise_vec[idx:idx+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3

        # gravity (3)
        noise_vec[idx:idx+3] = noise_scales.gravity * noise_level
        idx += 3

        # dog_pos (12)
        noise_vec[idx:idx+12] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += 12

        # dog_vel (12)
        noise_vec[idx:idx+12] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        idx += 12

        # dog previous actions (12)
        noise_vec[idx:idx+12] = 0.
        idx += 12

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)


        # PD gains / default pos 必须覆盖全部 DOF（20）
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # helpful constants
        # 3 + 3 + 3 + 12 + 12 + 12 (+ 第二章白盒负载 9)
        self.num_dog_one_step_obs = int(getattr(self.cfg.env, "num_dog_one_step_observations", 45))
        self.num_arm_one_step_obs = 30   # 6 + 6 + 6 + 6

        # 第二章负载真值在特权观测中的位置：本体观测 + base_lin_vel(3) + disturbance(3) + 高度图之后
        _payload_start = self.num_one_step_obs + 3 + 3 + (187 if self.cfg.terrain.measure_heights else 0)
        self.payload_priv_slice = (
            slice(_payload_start, _payload_start + self.payload_priv_dim)
            if self.payload_priv_dim > 0 else None
        )




        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.dog_actions = torch.zeros(
            self.num_envs, self.num_dog_actions,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        # 底座顺应残差（联合 fine-tune）：叠加在冻结底座动作上的增量，默认 0
        self.dog_residual = torch.zeros(
            self.num_envs, self.num_dog_actions,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        # 残差网络观测维度：dog one-step obs(45) + arm cmd(6) + arm se3 error(6)
        self.num_base_residual_obs = self.num_dog_one_step_obs + 6 + 6
        self.last_dog_actions = torch.zeros(
            self.num_envs, self.num_dog_actions,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_dog_actions = torch.zeros(
            self.num_envs, self.num_dog_actions,
            dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_contact_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.base_contact_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.base_contact_termination_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.roll_termination_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.pitch_termination_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.fall_termination_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_base_height = torch.zeros(
            self.num_envs, device=self.device
        )


        ###
        # arm command: [l, p, y, roll, pitch, yaw]
        self.commands_arm = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.commands_arm_xyz_error_scale = torch.tensor(
        [[0.35, 0.35, 0.25]],
        device=self.device,
        dtype=torch.float,
        requires_grad=False
            )

        # 给 actor 看的版本，第一版可以和 commands_arm 一样
        self.commands_arm_obs = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )

        # 当前末端状态
        self.ee_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_quat[:, 3] = 1.0

        self.arm_pos_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.arm_ori_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_arm_pos_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_arm_ori_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.arm_d_pos_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.arm_d_ori_error = torch.zeros(self.num_envs, 3, device=self.device)

        # 第四章：解析逆解的名义关节角（热启动缓存）
        self.arm_q_ik = torch.zeros(self.num_envs, 6, device=self.device)
        self.arm_target_vel = torch.zeros(
            self.num_envs, 6, device=self.device
        )

        # 末端位姿 FK 缓存标志（见 _refresh_ee_state）
        self._ee_state_fresh = False

        # --- 事件驱动 settle + PBRS buffers ---
        # 欧氏位置误差 [m] / 姿态误差 [rad]（区别于上面 lpy/abg 口径的误差）
        self.arm_pos_err_norm = torch.zeros(self.num_envs, device=self.device)
        self.arm_ori_err_norm = torch.zeros(self.num_envs, device=self.device)
        self.arm_settle_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.settle_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # PBRS potential: 上一步末尾（resample/reset 后）刷新的 phi
        self.arm_potential = torch.zeros(self.num_envs, device=self.device)
        # 本步 reward 前计算的 phi（与本步命令同口径）
        self.arm_potential_now = torch.zeros(self.num_envs, device=self.device)
        # 命令在 reward 之前被换掉的 env，本步 PBRS 置零
        self.arm_pbrs_skip = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- 容差课程状态 ---
        self.settle_pos_tol = float(self.cfg.arm.settle.pos_tol)
        self.settle_ori_tol = float(self.cfg.arm.settle.ori_tol)
        # 课程统计：累计 settle 事件数 / 步数（GPU 标量，避免每步同步）
        self._settle_events_acc = torch.zeros((), device=self.device)
        self._pos_intol_acc = torch.zeros((), device=self.device)
        self._ori_intol_acc = torch.zeros((), device=self.device)
        self._settle_steps_acc = 0
        self._last_settle_curriculum_it = -1

        # --- episode 级日志 ---
        self.episode_pos_err_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_ori_err_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_settle_count = torch.zeros(self.num_envs, device=self.device)

        # 可视化 / reward 用
        self.target_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.visual_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.obj_quats = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.obj_quats[:, 3] = 1.0



        self.commands_arm_lpy_range = torch.tensor(
            [
                self.cfg.arm.commands.l[1] - self.cfg.arm.commands.l[0],
                self.cfg.arm.commands.p[1] - self.cfg.arm.commands.p[0],
                self.cfg.arm.commands.y[1] - self.cfg.arm.commands.y[0],
            ],
            device=self.device,
            dtype=torch.float,
            requires_grad=False
            ).reshape(1, -1)
        
        self.commands_arm_rpy_range = torch.tensor(
            [
                self.cfg.arm.commands.roll_ee[1] - self.cfg.arm.commands.roll_ee[0],
                self.cfg.arm.commands.pitch_ee[1] - self.cfg.arm.commands.pitch_ee[0],
                self.cfg.arm.commands.yaw_ee[1] - self.cfg.arm.commands.yaw_ee[0],
            ],
            device=self.device,
            dtype=torch.float,
            requires_grad=False
            ).reshape(1, -1)

        self.commands_arm_lpy_range = torch.clamp(self.commands_arm_lpy_range, min=1e-6)
        self.commands_arm_rpy_range = torch.clamp(self.commands_arm_rpy_range, min=1e-6)

        self.target_abg = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        ##
        # -------------------------
        # arm URDF kinematics constants
        # origin = arm mount(base_link), no TCP offset
        # -------------------------
        self.arm_mount_xyz = to_torch([0.0, 0.0, 0.057], device=self.device)

        self.arm_joint_origin_xyz = to_torch([
            [0.0,     0.0,    0.0605],   # zarx_j1
            [0.02,    0.0,    0.04],     # zarx_j2
            [-0.264,  0.0,    0.0],      # zarx_j3
            [0.245,   0.0,   -0.056],    # zarx_j4
            [0.06675, 0.0,   -0.084],    # zarx_j5
            [0.03045, 0.0,    0.084],    # zarx_j6
        ], device=self.device).float()

        self.arm_joint_origin_rpy = to_torch([
            [0.0,       0.0, 0.0],       # zarx_j1
            [0.0,       0.0, 0.0],       # zarx_j2
            [math.pi,   0.0, 0.0],       # zarx_j3
            [0.0,       0.0, 0.0],       # zarx_j4
            [0.0,       0.0, 0.0],       # zarx_j5
            [-math.pi,  0.0, 0.0],       # zarx_j6
        ], device=self.device).float()

        self.arm_joint_axes = to_torch([
            [0.0, 0.0, 1.0],   # zarx_j1
            [0.0, 1.0, 0.0],   # zarx_j2
            [0.0, 1.0, 0.0],   # zarx_j3
            [0.0, 1.0, 0.0],   # zarx_j4
            [0.0, 0.0, 1.0],   # zarx_j5
            [1.0, 0.0, 0.0],   # zarx_j6
        ], device=self.device).float()

        # sim读出来的ee，仅用于debug对比
        self.ee_pos_sim = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_quat_sim = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_quat_sim[:, 3] = 1.0

        # 从 IK 可达性库加载关节配置，用于采样"按构造可达"的位姿命令
        self._load_ik_lib()
        tc = self.cfg.arm.trajectory_curriculum
        self.arm_trajectory_curriculum = ArmTrajectoryCurriculum(
            self.num_envs,
            self.device,
            enabled=tc.enabled,
            min_iterations=tc.min_iterations,
            min_success_rate=tc.min_success_rate,
            min_stability_rate=tc.min_stability_rate,
            evaluation_window=tc.evaluation_window,
            max_stage=tc.max_stage,
            load_position_coef=tc.load_position_coef,
            load_torque_coef=tc.load_torque_coef,
            load_torque_rate_coef=tc.load_torque_rate_coef,
        )


        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])


        # --- NEW: for touchdown impact & slip ---
        self.prev_contact_mask = torch.zeros(self.num_envs, len(self.feet_indices),
                                            dtype=torch.bool, device=self.device, requires_grad=False)

        # --- NEW: for angular acceleration penalty ---
        self.last_base_ang_vel = self.base_ang_vel.clone()  # (N,3)

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.arm_target_q = torch.zeros(self.num_envs, self.num_arm_dofs, device=self.device)





        dog_keys = sorted(list(self.cfg.control.stiffness.keys()), key=len, reverse=True)
        arm_keys = sorted(list(self.cfg.control.stiffness_arm.keys()), key=len, reverse=True)

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

            found = False
            if i < self.num_dog_actions:  # 狗
                for k in dog_keys:
                    if k in name:
                        self.p_gains[i] = self.cfg.control.stiffness[k]
                        self.d_gains[i] = self.cfg.control.damping[k]
                        found = True
                        break
            else:  # 臂
                for k in arm_keys:
                    if k in name:
                        self.p_gains[i] = self.cfg.control.stiffness_arm[k]
                        self.d_gains[i] = self.cfg.control.damping_arm[k]
                        found = True
                        break

            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.p_dog = self.p_gains[self.dog_dof_slice]
        self.p_arm = self.p_gains[self.arm_dof_slice]
        self.d_dog = self.d_gains[self.dog_dof_slice]
        self.d_arm = self.d_gains[self.arm_dof_slice]










        # 方便计算幅度：默认臂关节（env顺序）
        self.arm_q_default = self.default_dof_pos[0, self.arm_dof_slice].clone()  # (8,)



        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.obs_history = deque(maxlen=self.cfg.env.num_hist)
        for _ in range(self.cfg.env.num_hist):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_one_step_observations,dtype=torch.float, device=self.device))
        self.configure_base_policy_observation(
            getattr(
                self.cfg.env,
                "base_policy_history_len",
                self.cfg.env.num_hist,
            ),
            getattr(self.cfg.env, "base_policy_obs_dim", 45),
        )


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        # 奖励项可用 "组名__函数名" 归组（如 stab__orientation）；无前缀的一律算 task，
        # 因此第三、四章的现有配置无需改动
        group_weights = getattr(
            self.cfg.rewards,
            "single_critic_group_weights",
            (0.7, 0.2, 0.1),
        )
        if len(group_weights) != len(self.reward_groups):
            raise ValueError(
                "single_critic_group_weights must match reward groups"
            )
        self.reward_group_weights = torch.tensor(
            group_weights,
            dtype=torch.float,
            device=self.device,
        )
        if (
            (self.reward_group_weights < 0).any()
            or self.reward_group_weights.sum() <= 0
        ):
            raise ValueError(
                "single_critic_group_weights must be non-negative "
                "with a positive sum"
            )
        self.reward_group_weights /= self.reward_group_weights.sum()
        self.reward_functions = []
        self.reward_names = []
        self.reward_group_idx = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            base = name
            group = "task"
            if "__" in name:
                group, base = name.split("__", 1)
            if group not in self.reward_groups:
                raise ValueError(f"奖励 '{name}' 的组名 '{group}' 不在 {self.reward_groups} 中")
            g_idx = self.reward_groups.index(group)
            self.reward_names.append(name)
            self.reward_group_idx.append(g_idx)
            self.reward_functions.append(getattr(self, '_reward_' + base))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

        # 多评论家：分组奖励缓冲与分组 episode 统计
        self.rew_buf_vec = torch.zeros(self.num_envs, self.value_dim, dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.episode_sums_group = {
            g: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for g in self.reward_groups
        }

        # 多评论家下空组会让对应的 critic 恒学 V=0、该路 advantage 恒为 0，
        # 等于白白多了一个头却没起作用——这种配置几乎肯定是忘了给奖励加组前缀
        if self.value_dim > 1:
            empty_groups = []
            for gi, gname in enumerate(self.reward_groups[:self.value_dim]):
                if gi not in self.reward_group_idx:
                    empty_groups.append(gname)
            if empty_groups:
                raise ValueError(
                    "Multi-critic reward groups must be non-empty; missing "
                    f"{empty_groups}. Prefix rewards with '<group>__<name>'."
                )

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print(self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.foot_names = list(feet_names)

        ee_name = self.cfg.asset.ee_name   # 配置里加，比如 "gripper_link" / "tool0"
        ee_candidates = [i for i, n in enumerate(body_names) if ee_name in n]
        if len(ee_candidates) == 0:
            raise RuntimeError(f"Cannot find end-effector body with keyword: {ee_name}")

        self.ee_body_index = ee_candidates[0]
        print(f"[EE] using body index {self.ee_body_index}, name={body_names[self.ee_body_index]}")

        # 第二章负载感知：机械臂各连杆的 body 索引，用于算等效负载状态
        arm_kw = getattr(self.cfg.asset, "arm_body_keyword", "zarx")
        self.arm_body_indices = torch.tensor(
            [i for i, n in enumerate(body_names) if arm_kw in n],
            dtype=torch.long, device=self.device
        )
        if self.payload_awareness != "none" and self.arm_body_indices.numel() == 0:
            raise RuntimeError(f"负载感知已开启但找不到臂连杆，关键字: {arm_kw}")


        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
        if hasattr(self.cfg.terrain, "flat_warmup_iters"):
            flat_level = getattr(self.cfg.terrain, "flat_warmup_level", 0)

            self.terrain_levels[:] = flat_level
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels,
                self.terrain_types
            ]

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        has_env_ids = env_ids is not None and len(env_ids) > 0
        if self.cfg.terrain.mesh_type == 'plane':
            if has_env_ids:
                return torch.zeros(len(env_ids), self.num_height_points, device=self.device, requires_grad=False)
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if has_env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
            num_samples = len(env_ids)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
            num_samples = self.num_envs


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(num_samples, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        has_env_ids = env_ids is not None and len(env_ids) > 0
        if self.cfg.terrain.mesh_type == 'plane':
            if has_env_ids:
                return self.root_states[env_ids, 2].clone()
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if has_env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
            root_height = self.root_states[env_ids, 2]
            num_samples = len(env_ids)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)
            root_height = self.root_states[:, 2]
            num_samples = self.num_envs


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(num_samples, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(root_height.unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        has_env_ids = env_ids is not None and len(env_ids) > 0
        if self.cfg.terrain.mesh_type == 'plane':
            if has_env_ids:
                return self.feet_pos[env_ids, :, 2].clone()
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if has_env_ids:
            points = self.feet_pos[env_ids].clone()
            feet_pos_z = self.feet_pos[env_ids, :, 2]
            num_samples = len(env_ids)
        else:
            points = self.feet_pos.clone()
            feet_pos_z = self.feet_pos[:, :, 2]
            num_samples = self.num_envs

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(num_samples, -1) * self.terrain.cfg.vertical_scale

        feet_height = feet_pos_z - heights

        return feet_height

    #------------ reward functions----------------
    def _reward_base_residual_penalty(self):
        # 惩罚底座顺应残差的幅度：残差只在确实提升 manip 奖励时才偏离 0，
        # 否则被拉回，避免无意义的大幅乱动 / 把底座蹲到危险低姿态
        return torch.sum(torch.square(self.dog_residual), dim=1)

    def _reward_arm_manip_commands_tracking_combine(self):
        env_ids = torch.arange(self.num_envs, device=self.device)

        lpy = self.get_lpy_in_base_coord(env_ids)
        lpy_error = torch.sum(
            torch.abs(lpy - self.commands_arm_obs[:, 0:3]) / self.commands_arm_lpy_range,
            dim=1
        )

        # 姿态误差用四元数测地距离（rad），归一化后与位置项量级可比。
        # 旧的 abg 分量差度量有奇异性，会把真实 ~0.1rad 误差虚报成 ~0.2 且无法压低。
        cur_quat = self._ee_quat_in_yaw_base(env_ids)
        ori_err = self._quat_geodesic(cur_quat, self.target_ee_quat)
        rpy_error = ori_err / self.cfg.rewards.manip_ori_norm

        return torch.exp(
            -(self.cfg.rewards.manip_weight_lpy * lpy_error
            + self.cfg.rewards.manip_weight_rpy * rpy_error)
        )

    def _reward_arm_settle(self):
        # 一次性稀疏 bonus：误差连续 hold_steps 步在容差内（settle 当步发放，
        # 随后立即重采样新目标，所以每个 settle 事件只发一次）
        return self.settle_buf.float()

    def _reward_arm_pbrs_progress(self):
        # potential-based shaping: gamma*phi(s') - phi(s)，理论上不改变最优策略
        # phi = -(pos_err [m] + w * ori_err [rad])
        rew = (
            self.cfg.arm.settle.pbrs_gamma * self.arm_potential_now
            - self.arm_potential
        )
        # 命令在 reward 之前被换掉（定时重采样/初始 reset）的 env 本步跳过
        return torch.where(self.arm_pbrs_skip, torch.zeros_like(rew), rew)

    def _reward_arm_base_coupling(self):
        # 惩罚臂的关节加速度与底座角速度突变的同步性：
        # 臂可以动得快，但不能以激起底座振荡的方式动
        arm_acc = (
            self.dof_vel[:, self.arm_control_dof_slice]
            - self.last_dof_vel[:, self.arm_control_dof_slice]
        ) / self.dt
        arm_acc_norm = torch.norm(arm_acc, dim=1)

        base_ang_jump = torch.norm(self.base_ang_vel - self.last_base_ang_vel, dim=1)

        return arm_acc_norm * base_ang_jump

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = self._get_base_heights()
        return torch.square(
            base_height - self.cfg.rewards.base_height_target
        )

    def _reward_feet_slip(self):
        contact = self.contact_mask.float()
        slip = torch.sum(
            torch.square(self.feet_vel[:, :, :2]), dim=-1
        )
        return torch.sum(slip * contact, dim=1) / contact.sum(
            dim=1
        ).clamp_min(1.0)

    def _reward_joint_power(self):
        power = (
            self.torques[:, self.dog_dof_slice]
            * self.dof_vel[:, self.dog_dof_slice]
        )
        return torch.sum(torch.square(power), dim=1)

    def _reward_dof_acc(self):
        acceleration = (
            self.dof_vel[:, self.dog_dof_slice]
            - self.last_dof_vel[:, self.dog_dof_slice]
        ) / self.dt
        return torch.sum(torch.square(acceleration), dim=1)

    def _reward_action_rate(self):
        return torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )

    def _reward_base_contact(self):
        """Indicate sustained-contact onset on the base or other hard bodies."""
        return self.base_contact_buf.float()

    def _reward_foot_spacing(self):
        """Penalize a compact support polygon and insufficient inter-foot room."""
        slots = {
            name.split("_", 1)[0]: index
            for index, name in enumerate(self.foot_names)
        }
        feet_xy = self.feet_pos[:, :, :2]
        front_width = torch.norm(
            feet_xy[:, slots["FL"]] - feet_xy[:, slots["FR"]], dim=-1
        )
        rear_width = torch.norm(
            feet_xy[:, slots["RL"]] - feet_xy[:, slots["RR"]], dim=-1
        )
        pairs = (
            ("FL", "FR"),
            ("FL", "RL"),
            ("FL", "RR"),
            ("FR", "RL"),
            ("FR", "RR"),
            ("RL", "RR"),
        )
        minimum_distance = torch.stack(
            [
                torch.norm(
                    feet_xy[:, slots[first]] - feet_xy[:, slots[second]],
                    dim=-1,
                )
                for first, second in pairs
            ],
            dim=-1,
        ).min(dim=-1).values
        rewards_cfg = self.cfg.rewards
        target_width = float(rewards_cfg.foot_spacing_target_width)
        target_minimum = float(
            rewards_cfg.foot_spacing_target_min_distance
        )
        # Convert metres to normalized shortfall so the scale is independent of
        # the selected target.
        front_deficit = torch.relu(
            target_width - front_width
        ) / target_width
        rear_deficit = torch.relu(
            target_width - rear_width
        ) / target_width
        minimum_deficit = torch.relu(
            target_minimum - minimum_distance
        ) / target_minimum
        # Return a positive shortfall; the configured negative stability scale
        # converts this into a penalty.
        return (
            front_deficit.pow(2)
            + rear_deficit.pow(2)
            + minimum_deficit.pow(2)
        ) / 3.0

    def _reward_termination(self):
        """Terminal indicator for physical failures, excluding pure timeouts."""
        return (
            self.reset_buf.float()
            * (~self.time_out_buf).float()
        )

    def _reward_smoothness(self):
        return torch.sum(
            torch.square(
                self.actions
                - 2.0 * self.last_actions
                + self.last_last_actions
            ),
            dim=1,
        )



    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_arm_energy(self):
        arm_power_sq = torch.square(
            self.torques[:, self.arm_control_dof_slice] * self.dof_vel[:, self.arm_control_dof_slice]
        )
        return torch.sum(arm_power_sq, dim=1)
    
    def _reward_arm_dof_vel(self):
        return torch.sum(
            torch.square(self.dof_vel[:, self.arm_control_dof_slice]),
            dim=1
        )


    def _reward_arm_dof_acc(self):
        arm_acc = (
            self.dof_vel[:, self.arm_control_dof_slice]
            - self.last_dof_vel[:, self.arm_control_dof_slice]
        ) / self.dt
        return torch.sum(torch.square(arm_acc), dim=1)

    def _reward_arm_action_rate(self):
        return torch.sum(
            torch.square(self.last_actions - self.actions),
            dim=1
        )

    def _reward_arm_smoothness(self):
        return torch.sum(
            torch.square(self.actions - 2 * self.last_actions + self.last_last_actions),
            dim=1
        )


    def _reward_arm_pos_tracking_coarse(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        lpy = self.get_lpy_in_base_coord(env_ids)
        err = torch.norm(lpy - self.commands_arm_obs[:, 0:3], dim=1)
        return torch.exp(-err / 0.15)

    def _reward_arm_pos_tracking_fine(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        lpy = self.get_lpy_in_base_coord(env_ids)
        err = torch.norm(lpy - self.commands_arm_obs[:, 0:3], dim=1)
        return torch.exp(-err / 0.03)

    def _reward_arm_ori_tracking_coarse(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        err = self._quat_geodesic(self._ee_quat_in_yaw_base(env_ids), self.target_ee_quat)
        return torch.exp(-err / 0.5)

    def _reward_arm_ori_tracking_fine(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        err = self._quat_geodesic(self._ee_quat_in_yaw_base(env_ids), self.target_ee_quat)
        return torch.exp(-err / 0.12)

    def _reward_arm_error_progress(self):
        prev = torch.norm(self.last_arm_pos_error, dim=1) + 0.3 * torch.norm(self.last_arm_ori_error, dim=1)
        curr = torch.norm(self.arm_pos_error, dim=1) + 0.3 * torch.norm(self.arm_ori_error, dim=1)
        return torch.clamp(prev - curr, min=-0.05, max=0.05)
