import matplotlib.pyplot as plt
import sys
import time
import collections
import yaml
import onnxruntime as ort
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import torch
import pandas as pd  # 新增：用于数据导出

# === NEW: 扰动核心配置（可通过接口修改）===
DISTURBANCE_FORCE = np.array([0.0, 0.0, 0.0])  # 降低默认力（2000N过大会直接翻倒，建议200N以内）
DISTURBANCE_TORQUE = np.array([0.0, 0.0, 0.0])
DISTURB_BODY_NAME = "base"
_disturbance_duration = 0.2  # 默认扰动持续0.5秒（核心可配置项）
_disturbance_start_time = None  # 扰动开始的仿真时间
_is_disturbing = False  # 是否正在施加扰动
_trigger_disturbance_at = 2.0  # 新增：定时触发扰动的仿真时间（1秒）
_disturbance_triggered = False  # 新增：标记是否已触发定时扰动

# === 新增：姿态角记录相关 ===
# 存储数据：时间戳、滚转角(roll)、俯仰角(pitch)、偏航角(yaw)、是否施加扰动
pose_data = {
    "time": [],
    "roll": [],
    "pitch": [],
    "yaw": [],
    "is_disturbing": []
}

class KeyboardCmd:
    def __init__(self):
        self.forward=False
        self.backward=False
        self.left=False
        self.right=False
        self.turn_left=False
        self.turn_right=False
        self.reset=False
        self.disturb=False  # 触发扰动（按下F键）
    def zero(self):
        self.forward=False
        self.backward=False
        self.left=False
        self.right=False
        self.turn_left=False
        self.turn_right=False
        self.reset=False
        self.disturb=False
    def __str__(self) -> str:
        s="Key: "
        if self.forward:
            s+="W "
        if self.backward:
            s+="S "
        if self.left:
            s+="A "
        if self.right:
            s+="D "
        if self.turn_left:
            s+="Q "
        if self.turn_right:
            s+="E "
        if self.reset:
            s+="R "
        if self.disturb:
            s+="F (Disturbance) "
        return s

class KeyboardCmdHandler:
    def __init__(self,max_linear_speed=5,max_angular_speed=1.5,mode_game=False,step=0.01):
        self.linear_x = 0
        self.linear_y = 0
        self.angular_z = 0
        self.heading = 0
        self.step=step
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.game_mode=mode_game
    def update(self,key_cmd:KeyboardCmd):
        if key_cmd.forward:
            self.linear_x += self.step
        elif self.game_mode:
            if self.linear_x>0:
                self.linear_x -= self.step
        if key_cmd.backward:
            self.linear_x -= self.step
        elif self.game_mode:
            if self.linear_x<0:
                self.linear_x += self.step
        if key_cmd.right:
            self.linear_y += self.step
        elif self.game_mode:
            if self.linear_y>0:
                self.linear_y -= self.step
        if key_cmd.left:
            self.linear_y -= self.step
        elif self.game_mode:
            if self.linear_y<0:
                self.linear_y += self.step
        if key_cmd.turn_left:
            self.angular_z += self.step
        elif self.game_mode:
            if self.angular_z>0:
                self.angular_z -= self.step
        if key_cmd.turn_right:
            self.angular_z -= self.step
        elif self.game_mode: 
            if self.angular_z<0:
                self.angular_z += self.step

        # clip values
        self.linear_x = np.clip(self.linear_x,-self.max_linear_speed,self.max_linear_speed)
        self.linear_y= np.clip(self.linear_y,-self.max_linear_speed,self.max_linear_speed)
        self.angular_z = np.clip(self.angular_z,-self.max_angular_speed,self.max_angular_speed)

# === 全局扰动控制接口 ===
kb_cmd = KeyboardCmd()

def set_disturbance_duration(duration: float):
    """
    【扰动时长控制接口】设置扰动持续时间（秒）
    参数：
        duration: 扰动持续时间（≥0，0表示仅施加1帧）
    """
    global _disturbance_duration
    if duration < 0:
        print(f"警告：扰动时长不能为负，已重置为0.5秒")
        _disturbance_duration = 0.5
    else:
        _disturbance_duration = duration
        print(f"扰动时长已设置为：{_disturbance_duration} 秒")

def stop_disturbance():
    """【扰动控制接口】立即停止当前施加的扰动"""
    global _is_disturbing, _disturbance_start_time
    _is_disturbing = False
    _disturbance_start_time = None
    print("扰动已手动停止")

def keyboard_callback(keycode):
        try:
            ch = chr(keycode).upper()
        except Exception:
            return
        print(f"key {ch} is pressed")
        global _is_disturbing, _disturbance_start_time, _disturbance_triggered
        
        if ch=='W':
            kb_cmd.forward=True
        elif ch=='S':
            kb_cmd.backward=True
        elif ch=='A':
            kb_cmd.left=True
        elif ch=='D':
            kb_cmd.right=True
        elif ch=='Q':
            kb_cmd.turn_left=True
        elif ch=='E':
            kb_cmd.turn_right=True
        elif ch=='R':
            kb_cmd.reset=True
            stop_disturbance()  # 重置时停止扰动
            _disturbance_triggered = False  # 重置定时触发标记
        elif ch=='F':
            # 按下F键触发扰动（记录开始时间，标记为正在扰动）
            _is_disturbing = True
            _disturbance_start_time = None  # 延迟到仿真循环中获取准确的仿真时间
            _disturbance_triggered = True  # 标记已触发
            print(f"已手动触发扰动（将持续 {_disturbance_duration} 秒）")
        elif ch=='G':
            # 按下G键立即停止扰动
            stop_disturbance()

# === 施加外力扰动的函数（支持持续施加）===
def apply_disturbance(m, d, body_name, force, torque=np.zeros(3)):
    """
    给指定Body施加外力/力矩（世界坐标系）
    参数：
        m: mujoco模型
        d: mujoco数据
        body_name: 目标Body名称（如"base"）
        force: 施加的力 [fx, fy, fz] (N)
        torque: 施加的力矩 [tx, ty, tz] (N·m)
    """
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        print(f"警告：未找到Body '{body_name}'，跳过扰动")
        return
    
    # 持续施加力/力矩（每帧更新，直到时长结束）
    d.xfrc_applied[body_id][0:3] = force
    d.xfrc_applied[body_id][3:6] = torque

def check_disturbance_status(d):
    """检查扰动是否应继续施加，返回是否需要停止"""
    global _is_disturbing, _disturbance_start_time, _disturbance_duration
    
    if not _is_disturbing:
        return False
    
    # 首次触发时，记录当前仿真时间
    if _disturbance_start_time is None:
        _disturbance_start_time = d.time
        print(f"扰动开始时间（仿真时间）：{_disturbance_start_time:.3f} 秒")
    
    # 计算已持续时间
    elapsed_time = d.time - _disturbance_start_time
    
    # 检查是否达到持续时长
    if elapsed_time >= _disturbance_duration:
        print(f"扰动结束（已持续 {elapsed_time:.3f} 秒，设定时长 {_disturbance_duration} 秒）")
        _is_disturbing = False
        _disturbance_start_time = None
        return True
    return False

# === 新增：从四元数计算欧拉角（roll/pitch/yaw）===
def quat_to_euler(quat):
    """
    四元数转欧拉角（XYZ顺序，单位：弧度）
    quat: [w, x, y, z]
    返回：roll(x), pitch(y), yaw(z)
    """
    w, x, y, z = quat
    # 滚转角 (x)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # 俯仰角 (y)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    
    # 偏航角 (z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

# === 新增：保存姿态角数据 ===
def save_pose_data(save_path="robot_pose_data.csv"):
    """将姿态角数据保存为CSV文件"""
    df = pd.DataFrame(pose_data)
    # 可选：将弧度转换为角度（更易读）
    df['roll_deg'] = np.degrees(df['roll'])
    df['pitch_deg'] = np.degrees(df['pitch'])
    df['yaw_deg'] = np.degrees(df['yaw'])
    df.to_csv(save_path, index=False)
    print(f"姿态角数据已保存到：{save_path}")
    return df

# === 修正：关节顺序转换函数（拆分两个方向）===
def mujoco2train(q):
    """
    Mujoco顺序（FR,FL,RR,RL）→ 训练顺序（FL,FR,RL,RR）
    q: 12维关节数组（每腿3个关节）
    """
    q_copy = q.copy()  # 深拷贝避免原数组被修改
    train_q = np.zeros_like(q_copy)
    # FL: Mujoco[3-5] → Train[0-2]
    train_q[0:3] = q_copy[3:6]
    # FR: Mujoco[0-2] → Train[3-5]
    train_q[3:6] = q_copy[0:3]
    # RL: Mujoco[9-12] → Train[6-8]
    train_q[6:9] = q_copy[9:12] 
    # RR: Mujoco[6-9] → Train[9-12]
    train_q[9:12] = q_copy[6:9] 
    return train_q

def train2mujoco(q):
    """
    训练顺序（FL,FR,RL,RR）→ Mujoco顺序（FR,FL,RR,RL）
    q: 12维关节数组（每腿3个关节）
    """
    q_copy = q.copy()  # 深拷贝避免原数组被修改
    mujoco_q = np.zeros_like(q_copy)
    # FR: Train[3-5] → Mujoco[0-2]
    mujoco_q[0:3] = q_copy[3:6]
    # FL: Train[0-2] → Mujoco[3-5]
    mujoco_q[3:6] = q_copy[0:3]
    # RR: Train[9-12] → Mujoco[6-8]
    mujoco_q[6:9] = q_copy[9:12] if len(q_copy)>=12 else q_copy[6:9]
    # RL: Train[6-8] → Mujoco[9-12]
    mujoco_q[9:12] = q_copy[6:9] if len(q_copy)>=12 else np.zeros(3)
    return mujoco_q

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    for path_key in ['policy_path', 'xml_path']:
        config[path_key] = config[path_key]
    
    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)

    # 修正：default_angles从训练顺序转换为Mujoco顺序
    if len(config['default_angles']) == 12:
        # config['default_angles'] = train2mujoco(config['default_angles'])
        print("default_angles已转换为Mujoco关节顺序")

    config['PLOT'] = bool(config.get('PLOT', False))
    config['plot_decimation'] = int(config.get('plot_decimation', 1))
    
    return config

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def compute_observation(d, config, action, cmd, n_joints,phase):
    """Compute the observation vector from current state"""
    print("height is",d.qpos[2])
    # 读取Mujoco关节位置/速度 → 转换为训练顺序
    qj = d.qpos[7:7+n_joints].copy()
    qj = mujoco2train(qj)  # 替换原trans函数，使用正确的Mujoco→训练转换
    dqj = d.qvel[6:6+n_joints].copy()
    dqj = mujoco2train(dqj)  # 替换原trans函数
    
    quat = d.qpos[3:7].copy()
    omega_world = d.qvel[3:6].copy()
    omega_body = quat_rotate_inverse(quat,omega_world)
    # if len(config['default_angles']) < n_joints:
    #     padded_defaults = np.zeros(n_joints, dtype=np.float32)
    #     padded_defaults[:len(config['default_angles'])] = config['default_angles']
    # else:
    padded_defaults = config['default_angles'][:n_joints]
    padded_defaults = mujoco2train(padded_defaults)
    
    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega_world * config['ang_vel_scale']
    
    # 计算stand_cmd
    cmd_vec = np.array([float(cmd.vx), float(cmd.vy), float(cmd.dyaw)], dtype=np.float32)
    cmd_norm = np.linalg.norm(cmd_vec)
    stand_cmd_threshold = config.get('stand_cmd_threshold', 0.01)
    stand_cmd = 1.0 if cmd_norm <= stand_cmd_threshold else 0.0
    
    single_obs_dim = 3  + 3 + 3 + n_joints + n_joints + n_joints
    
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd_vec * config['cmd_scale']
    # single_obs[3] = np.sin(phase * 2 * np.pi)
    # single_obs[4] = np.cos(phase * 2 * np.pi)
    single_obs[3:6] = omega_scaled
    single_obs[6:9] = gravity_orientation
    single_obs[9:9+n_joints] = qj_scaled
    single_obs[9+n_joints:9+2*n_joints] = dqj_scaled
    single_obs[9+2*n_joints:9+3*n_joints] = action

    return single_obs, single_obs_dim

def _low_pass_action_filter(actions,last_actions,flt):
    actons_filtered = actions * flt + last_actions * (1-flt)
    return actons_filtered

def exp_avg_filter( x, avg, alpha=0.95):
    """Simple exponential average filter"""
    avg = alpha*x + (1-alpha)*avg
    return avg

# === PLOT ===
def plot_joint_curves(times, q_hist, dq_hist, joint_indices=(6, 9), labels=None):
    """Plot joint position/velocity curves for specified joints"""
    times = np.asarray(times)
    q_hist = np.asarray(q_hist)
    dq_hist = np.asarray(dq_hist)
    n_joints = q_hist.shape[1]

    if labels is None:
        labels = {6: "Right_hip_pitch ", 9: "Right_knee"}

    valid_idxs = []
    for idx in joint_indices:
        if 0 <= idx < n_joints:
            valid_idxs.append(idx)
        else:
            print(f"[WARN] joint index {idx} out of range (n_joints={n_joints}), skipped.")

    if not valid_idxs:
        print("[WARN] No valid joint indices to plot.")
        return

    # Positions
    plt.figure()
    for idx in valid_idxs:
        lbl = labels.get(idx, f"J{idx}")
        plt.plot(times, q_hist[:, idx], label=lbl)
    plt.title('Joint Position (rad)',fontsize=12)
    plt.xlabel('Time (s)',fontsize=12)
    plt.ylabel('q (rad)',fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Velocities
    plt.figure(figsize=(12,5))
    for idx in valid_idxs:
        lbl = labels.get(idx, f"J{idx}")
        plt.plot(times, dq_hist[:, idx], label=lbl)
    plt.xlabel('Time (s)',fontsize=12)
    plt.ylabel('Joint Velocity (rad/s)',fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.show()

# === 新增：绘制姿态角变化曲线（可选）===
def plot_pose_curves(df):
    """绘制姿态角随时间变化曲线"""
    plt.figure(figsize=(12, 8))
    
    # 绘制弧度曲线
    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['roll'], label='Roll (rad)', color='r')
    plt.plot(df['time'], df['pitch'], label='Pitch (rad)', color='g')
    plt.plot(df['time'], df['yaw'], label='Yaw (rad)', color='b')
    plt.axvline(x=_trigger_disturbance_at, color='k', linestyle='--', label='Disturbance Start (1s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Euler Angle (rad)')
    plt.title('Robot Pose (Euler Angles) vs Time')
    plt.legend()
    plt.grid(True)
    
    # 绘制角度曲线（更易读）
    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df['roll_deg'], label='Roll (°)', color='r')
    plt.plot(df['time'], df['pitch_deg'], label='Pitch (°)', color='g')
    plt.plot(df['time'], df['yaw_deg'], label='Yaw (°)', color='b')
    plt.axvline(x=_trigger_disturbance_at, color='k', linestyle='--', label='Disturbance Start (1s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Euler Angle (°)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("robot_pose_disturbance.png")
    plt.show()

def main():
    # ====================== 扰动配置 =======================
    set_disturbance_duration(0.2)  # 扰动持续0.2秒
    print(f"将在仿真时间 {_trigger_disturbance_at} 秒时自动触发扰动")
    # =======================================================

    config_path = Path(__file__).resolve().parent / 'configs' / 'go2.yaml'
    config = load_config(str(config_path))
    
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    n_joints = config['num_actions']

    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = m.actuator_trnid[i]  # (joint_id, dof_id)
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, trnid[0])
        print(i, name, "->", jname)
    
    # Initialize variables
    action = np.zeros(config['num_actions'], dtype=np.float32)
    action_flt = np.zeros(config['num_actions'], dtype=np.float32)
    action_scaled = np.zeros(config['num_actions'], dtype=np.float32)
    # default_angles已在load_config中转换为Mujoco顺序，无需再次转换
    target_dof_pos = config['default_angles'].copy()
  
    # Initialize observation history
    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, n_joints,phase=0)
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    
    obs = np.zeros(config['num_obs'], dtype=np.float32)
    
    # Load ONNX policy
    ort_session = ort.InferenceSession(config['policy_path'], providers=['CPUExecutionProvider'])
    ort_input_name = ort_session.get_inputs()[0].name
    
    counter = 0
    action_mujoco = np.zeros(12)
    # Plot buffers
    times = []
    q_hist = []
    dq_hist = []
    decim = max(1, int(config['plot_decimation']))
    step_counter_for_plot = 0
    render_stride = max(1, int(round(1.0 / (m.opt.timestep * max(1, 50)))))
    delay_steps = 0
    # 修正：action_delay_buffer初始值使用Mujoco顺序的default_angles
    action_delay_buffer = collections.deque(
        [config['default_angles'].copy() for _ in range(delay_steps)], 
        maxlen=delay_steps
    )

    # Phase variables
    cycle_time = config.get('cycle_time', 0.6)
    phase_length_buf = 0

    keyboard_cmd_handler = KeyboardCmdHandler(step=0.1)
    with mujoco.viewer.launch_passive(m, d,key_callback=keyboard_callback) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()
            
            # ====================== 1. 定时触发扰动 ======================
            global _disturbance_triggered, _is_disturbing
            # 仿真时间达到1秒且未触发过扰动 → 自动触发
            if d.time >= _trigger_disturbance_at and not _disturbance_triggered:
                _is_disturbing = True
                _disturbance_start_time = None  # 延迟到check_disturbance_status中记录
                _disturbance_triggered = True
                print(f"定时扰动触发！当前仿真时间：{d.time:.3f} 秒")
            # =============================================================
            
            # ====================== 2. 施加/停止扰动 ======================
            if _is_disturbing:
                apply_disturbance(m, d, DISTURB_BODY_NAME, DISTURBANCE_FORCE, DISTURBANCE_TORQUE)
                # 检查是否达到持续时长，若达到则停止
                if check_disturbance_status(d):
                    # 清空外力（停止后重置xfrc_applied）
                    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, DISTURB_BODY_NAME)
                    if body_id != -1:
                        d.xfrc_applied[body_id] = np.zeros(6)
            # =============================================================
            
            # ====================== 3. 记录姿态角 ======================
            # 获取基座四元数（mujoco中qpos[3:7]是[w,x,y,z]）
            base_quat = d.qpos[3:7].copy()
            roll, pitch, yaw = quat_to_euler(base_quat)
            # 存储数据
            pose_data['time'].append(d.time)
            pose_data['roll'].append(roll)
            pose_data['pitch'].append(pitch)
            pose_data['yaw'].append(yaw)
            pose_data['is_disturbing'].append(_is_disturbing)
            # =============================================================
            
            # 重置机器人逻辑
            if kb_cmd.reset:
                mujoco.mj_resetData(m, d)
                # 修正：重置时target_dof_pos使用Mujoco顺序的default_angles
                target_dof_pos = config['default_angles'].copy()
                action = np.zeros(config['num_actions'])
                action_flt = np.zeros(config['num_actions'])
                phase_length_buf = 0
                keyboard_cmd_handler.linear_x = 0
                keyboard_cmd_handler.linear_y = 0
                keyboard_cmd_handler.angular_z = 0
                print("机器人已重置！")
                kb_cmd.reset = False
                # 重置扰动标记和姿态数据
                _disturbance_triggered = False
                _is_disturbing = False
                pose_data['time'].clear()
                pose_data['roll'].clear()
                pose_data['pitch'].clear()
                pose_data['yaw'].clear()
                pose_data['is_disturbing'].clear()
                obs_history.clear()
                obs = np.zeros(config['num_obs'], dtype=np.float32)

            # Control leg joints with policy
            action = np.clip(action, -100, 100)
            # 修正：action是训练顺序 → 转换为Mujoco顺序
            # action_mujoco = train2mujoco(action)
            # 基于Mujoco顺序的action计算scaled/flt
            action_scaled = action_mujoco * config['action_scales'] 
            action_flt = exp_avg_filter(action_scaled, action_flt, 1)
            # target_dof_pos是Mujoco顺序，直接累加
            target_dof_pos = action_flt + config['default_angles']

            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+config['num_actions']],  # Mujoco顺序的关节位置
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+config['num_actions']],  # Mujoco顺序的关节速度
                config['kds']
            )

            d.ctrl[:config['num_actions']] = leg_tau
            
            # Keep other joints at zero positions if they exist
            if n_joints > config['num_actions']:
                arm_kp = 100.0
                arm_kd = 0.5
                arm_target_positions = np.zeros(n_joints - config['num_actions'], dtype=np.float32)
                
                arm_tau = pd_control(
                    arm_target_positions,
                    d.qpos[7+config['num_actions']:7+n_joints],
                    np.ones(n_joints-config['num_actions']) * arm_kp,
                    np.zeros(n_joints-config['num_actions']),
                    d.qvel[6+config['num_actions']:6+n_joints],
                    np.ones(n_joints-config['num_actions']) * arm_kd
                )
                
                if d.ctrl.shape[0] > config['num_actions']:
                    d.ctrl[config['num_actions']:] = arm_tau
            
            # Step physics
            mujoco.mj_step(m, d)

            # Record for plotting
            if config['PLOT']:
                step_counter_for_plot += 1
                if step_counter_for_plot % decim == 0:
                    times.append(d.time)
                    q_hist.append(d.qpos[7:7+n_joints].copy())
                    dq_hist.append(d.qvel[6:6+n_joints].copy())
            
            counter += 1
            if counter % config['control_decimation'] == 0:
                # Update observation
                keyboard_cmd_handler.update(kb_cmd)
                # cmd.vx = keyboard_cmd_handler.linear_x
                cmd.vx = 1
                cmd.vy = keyboard_cmd_handler.linear_y
                cmd.dyaw = keyboard_cmd_handler.angular_z

                print(f"cmd_vx: {cmd.vx:.3f}, cmd_vy: {cmd.vy:.3f}, cmd_dyaw: {cmd.dyaw:.3f}")

                kb_cmd.zero()
                # Update phase
                phase_length_buf += 1
                phase = (phase_length_buf * 0.02 / cycle_time) % 1.0   
                # compute_observation中已自动转换关节顺序，无需额外处理
                single_obs, _ = compute_observation(d, config, action, cmd, n_joints,phase)
                single_obs = np.clip(single_obs, -100, 100)
                obs_history.append(single_obs)
                
                # Construct full observation with history
                # for i, hist_obs in enumerate(reversed(obs_history)):
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs
                
                # Policy inference (ONNX) → 输出是训练顺序的action
                input_batch = obs.astype(np.float32)[None, :]
                onnx_outputs = ort_session.run(None, {ort_input_name: input_batch})
                action = onnx_outputs[0].squeeze()  # 训练顺序，后续在控制逻辑中转换为Mujoco顺序
                action_mujoco = train2mujoco(action)
                # Sync viewer
                if (counter % render_stride) == 0:
                    viewer.sync()
            
            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # ====================== 仿真结束后：保存姿态数据 + 绘图 ======================
    if len(pose_data['time']) > 0:
        # 保存数据到CSV
        df = save_pose_data("robot_pose_disturbance_data_him.csv")
        # 可选：绘制姿态角曲线
        plot_pose_curves(df)
    
    # 原有关节曲线绘制
    if config['PLOT'] and len(times) > 1:
        try:
            plot_joint_curves(times, q_hist, dq_hist, joint_indices=(6, 9))
        except Exception as e:
            print(f"[WARN] Failed to plot curves: {e}")

if __name__ == "__main__":
    main()
