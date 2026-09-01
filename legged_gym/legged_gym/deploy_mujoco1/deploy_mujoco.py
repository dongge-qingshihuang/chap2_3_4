

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
# === NEW ===


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

class KeyboardCmd:
    def __init__(self):
        self.forward=False
        self.backward=False
        self.left=False
        self.right=False
        self.turn_left=False
        self.turn_right=False
        self.reset=False
    def zero(self):
        self.forward=False
        self.backward=False
        self.left=False
        self.right=False
        self.turn_left=False
        self.turn_right=False
        self.reset=False
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

kb_cmd = KeyboardCmd()

def keyboard_callback(keycode):
        # 一些系统发送小写；统一转大写兼容
        try:
            ch = chr(keycode).upper()
        except Exception:
            return
        print(f"key {ch} is pressed")
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

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0


def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process paths with LEGGED_GYM_ROOT_DIR
    for path_key in ['policy_path', 'xml_path']:
        config[path_key] = config[path_key]
    
    # Convert lists to numpy arrays where needed
    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)

    # === NEW === plotting flags (with defaults)
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


def compute_observation(d, config, action, cmd, n_joints):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    qj = d.qpos[7:7+n_joints].copy()
    qj = mujoco2train(qj)
    dqj = d.qvel[6:6+n_joints].copy()
    dqj = mujoco2train(dqj)
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    
    # Handle default angles padding
    if len(config['default_angles']) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(config['default_angles'])] = config['default_angles']
    else:
        padded_defaults = config['default_angles'][:n_joints]
        padded_defaults = mujoco2train(padded_defaults)
    
    # Scale the values
    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']

    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']
    
    # Calculate single observation dimension
    single_obs_dim = 3 + 3 + 3 + n_joints + n_joints + n_joints
    cmd_vec = np.array([float(cmd.vx), float(cmd.vy), float(cmd.dyaw)], dtype=np.float32)
    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd_vec * config['cmd_scale']
    single_obs[3:6] = omega_scaled
    single_obs[6:9] = gravity_orientation
    single_obs[9:9+n_joints] = qj_scaled
    single_obs[9+n_joints:9+2*n_joints] = dqj_scaled
    single_obs[9+2*n_joints:9+3*n_joints] = action
    
    return single_obs, single_obs_dim

def _low_pass_action_filter(actions,last_actions,flt):
    actons_filtered = actions * flt + last_actions * (1-flt)
    return actons_filtered

def exp_avg_filter( x, avg, alpha=0.8):
    """
    Simple exponential average filter
    """
    avg = alpha*x + (1-alpha)*avg
    return avg

# === PLOT ===
def plot_joint_curves(times, q_hist, dq_hist, joint_indices=(6, 9), labels=None):
    """
    仅绘制给定 joint_indices 的关节曲线（默认画索引 6 和 9）。
    times: [T]
    q_hist: [T, n_joints]
    dq_hist: [T, n_joints]
    joint_indices: 关节索引元组/列表，例如 (6, 9)
    labels: 可选的 {idx: name} 字典，用于图例命名
    """
    times = np.asarray(times)
    q_hist = np.asarray(q_hist)
    dq_hist = np.asarray(dq_hist)
    n_joints = q_hist.shape[1]

    if labels is None:
        labels = {6: "Right_hip_pitch ", 9: "Right_knee"}

    # 过滤掉越界索引
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
    # plt.title('Joint Velocity (rad/s)')
    plt.xlabel('Time (s)',fontsize=12)
    plt.ylabel('Joint Velocity (rad/s)',fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.show()



def main():
    # Load configuration from configs/g1.yaml next to this file
    config_path = Path(__file__).resolve().parent / 'configs' / 'go2.yaml'
    config = load_config(str(config_path))
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    # Check number of joints
    n_joints = config['num_actions']
   # print(f"Robot has {n_joints} joints in MuJoCo model")
    
    # Initialize variables
    action = np.zeros(config['num_actions'], dtype=np.float32)
    action_flt = np.zeros(config['num_actions'], dtype=np.float32)
    action_scaled = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    # cmd = config['cmd_init'].copy()
  
    # Initialize observation history
    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, n_joints)
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    
    # Prepare full observation vector
    obs = np.zeros(config['num_obs'], dtype=np.float32)
    
    # Load ONNX policy
    ort_session = ort.InferenceSession(config['policy_path'], providers=['CPUExecutionProvider'])
    ort_input_name = ort_session.get_inputs()[0].name

     # load policy
    # policy = torch.jit.load(config['policy_jit_path'])
    
    counter = 0

    # === NEW === buffers for plotting
    times = []
    q_hist = []
    dq_hist = []
    decim = max(1, int(config['plot_decimation']))
    step_counter_for_plot = 0
    render_stride = max(1, int(round(1.0 / (m.opt.timestep * max(1, 50)))))  # dt=0.001, 20 FPS -> 50步渲染一次
        # 初始化动作延迟队列（3步延迟）
    delay_steps = 1
    action_delay_buffer = collections.deque(
        [np.zeros_like(config['default_angles']) for _ in range(delay_steps)], 
        maxlen=delay_steps
    )

   
    keyboard_cmd_handler = KeyboardCmdHandler(step=0.1)
    with mujoco.viewer.launch_passive(m, d,key_callback=keyboard_callback) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()
            
            
            
            
            # Control leg joints with policy
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+config['num_actions']],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+config['num_actions']],
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

            # === NEW === record for plotting (after step so state is updated)
            if config['PLOT']:
                step_counter_for_plot += 1
                if step_counter_for_plot % decim == 0:
                    times.append(d.time)  # MuJoCo sim time in seconds
                    q_hist.append(d.qpos[7:7+n_joints].copy())
                    dq_hist.append(d.qvel[6:6+n_joints].copy())
            
            counter += 1
            if counter % config['control_decimation'] == 0:
                # Update observation
                keyboard_cmd_handler.update(kb_cmd)  # type: ignore
                cmd.vx = keyboard_cmd_handler.linear_x
                cmd.vy = keyboard_cmd_handler.linear_y
                cmd.dyaw = keyboard_cmd_handler.angular_z

                print(f"cmd_vx: {cmd.vx:.3f}, cmd_vy: {cmd.vy:.3f}, cmd_dyaw: {cmd.dyaw:.3f}")

                kb_cmd.zero()
                single_obs, _ = compute_observation(d, config, action, cmd, n_joints)
                single_obs = np.clip(single_obs, -100, 100)
                obs_history.append(single_obs)
                
                # Construct full observation with history
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs
                
                # Policy inference (ONNX) → 输出是训练顺序的action
                input_batch = obs.astype(np.float32)[None, :]
                onnx_outputs = ort_session.run(None, {ort_input_name: input_batch})
                action = onnx_outputs[0].squeeze()  # 训练顺序，后续在控制逻辑中转换为Mujoco顺序
                action_mujoco = train2mujoco(action)
                action = np.clip(action, -100, 100)
                action_scaled = action_mujoco * config['action_scales']
                action_flt = exp_avg_filter(action_scaled, action_flt, 1)

                # === 加入延迟队列 ===
                delayed_action = action_flt + config['default_angles']
                action_delay_buffer.append(delayed_action)   # 新动作入队
                target_dof_pos = action_delay_buffer[0]      # 取最早的动作作为当前控制输入

                
               
                # Sync viewer
                if (counter % render_stride) == 0:
                    viewer.sync()
            
                
            
            
            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # === NEW === Plot after viewer closes / simulation ends
    if config['PLOT'] and len(times) > 1:
        try:
            plot_joint_curves(times, q_hist, dq_hist, joint_indices=(6, 9))

        except Exception as e:
            print(f"[WARN] Failed to plot curves: {e}")


if __name__ == "__main__":
    main()
