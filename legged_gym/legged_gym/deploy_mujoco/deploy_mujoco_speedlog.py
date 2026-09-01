import time
import collections
import yaml
import onnxruntime as ort
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import pandas as pd


# =========================
# 输出CSV配置（直接改这里）
# =========================
SAVE_CSV = True
CSV_SAVE_DIR = "."  # 输出目录
CSV_NAME_PREFIX = "speed_tracking"  # 输出文件前缀
# =========================


# =========================
# 键盘指令
# =========================
class KeyboardCmd:
    def __init__(self):
        self.forward = False
        self.backward = False
        self.left = False
        self.right = False
        self.turn_left = False
        self.turn_right = False
        self.reset = False

    def zero(self):
        self.forward = False
        self.backward = False
        self.left = False
        self.right = False
        self.turn_left = False
        self.turn_right = False
        self.reset = False


class KeyboardCmdHandler:
    def __init__(self, max_linear_speed=5.0, max_angular_speed=1.5, mode_game=False, step=0.01):
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0
        self.heading = 0.0
        self.step = step
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.game_mode = mode_game

    def update(self, key_cmd: KeyboardCmd):
        if key_cmd.forward:
            self.linear_x += self.step
        elif self.game_mode and self.linear_x > 0:
            self.linear_x -= self.step

        if key_cmd.backward:
            self.linear_x -= self.step
        elif self.game_mode and self.linear_x < 0:
            self.linear_x += self.step

        if key_cmd.right:
            self.linear_y += self.step
        elif self.game_mode and self.linear_y > 0:
            self.linear_y -= self.step

        if key_cmd.left:
            self.linear_y -= self.step
        elif self.game_mode and self.linear_y < 0:
            self.linear_y += self.step

        if key_cmd.turn_left:
            self.angular_z += self.step
        elif self.game_mode and self.angular_z > 0:
            self.angular_z -= self.step

        if key_cmd.turn_right:
            self.angular_z -= self.step
        elif self.game_mode and self.angular_z < 0:
            self.angular_z += self.step

        self.linear_x = np.clip(self.linear_x, -self.max_linear_speed, self.max_linear_speed)
        self.linear_y = np.clip(self.linear_y, -self.max_linear_speed, self.max_linear_speed)
        self.angular_z = np.clip(self.angular_z, -self.max_angular_speed, self.max_angular_speed)


kb_cmd = KeyboardCmd()


def keyboard_callback(keycode):
    try:
        ch = chr(keycode).upper()
    except Exception:
        return

    if ch == 'W':
        kb_cmd.forward = True
    elif ch == 'S':
        kb_cmd.backward = True
    elif ch == 'A':
        kb_cmd.left = True
    elif ch == 'D':
        kb_cmd.right = True
    elif ch == 'Q':
        kb_cmd.turn_left = True
    elif ch == 'E':
        kb_cmd.turn_right = True
    elif ch == 'R':
        kb_cmd.reset = True


# =========================
# 关节顺序转换（保持你原来的逻辑）
# =========================
def mujoco2train(q):
    """
    Mujoco顺序（FR,FL,RR,RL）→ 训练顺序（FL,FR,RL,RR）
    q: 12维关节数组（每腿3个关节）
    """
    q_copy = q.copy()
    train_q = np.zeros_like(q_copy)
    train_q[0:3] = q_copy[3:6]    # FL
    train_q[3:6] = q_copy[0:3]    # FR
    train_q[6:9] = q_copy[9:12]   # RL
    train_q[9:12] = q_copy[6:9]   # RR
    return train_q


def train2mujoco(q):
    """
    训练顺序（FL,FR,RL,RR）→ Mujoco顺序（FR,FL,RR,RL）
    q: 12维关节数组（每腿3个关节）
    """
    q_copy = q.copy()
    mujoco_q = np.zeros_like(q_copy)
    mujoco_q[0:3] = q_copy[3:6]    # FR
    mujoco_q[3:6] = q_copy[0:3]    # FL
    mujoco_q[6:9] = q_copy[9:12]   # RR
    mujoco_q[9:12] = q_copy[6:9]   # RL
    return mujoco_q


# =========================
# 四元数与旋转：把世界系速度转到机体系
# MuJoCo free joint quaternion: [w, x, y, z]
# =========================
def quat_rotate_inverse(q, v):
    """
    用 q 的逆旋转向量 v（把世界系向量转到body系）
    q: [w,x,y,z]
    v: [3,]
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    qvec = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    # inverse rotation == rotate by conjugate (w, -qvec)
    qvec_inv = -qvec
    t = 2.0 * np.cross(qvec_inv, v)
    return v + w * t + np.cross(qvec_inv, t)


# =========================
# 配置加载
# =========================
class cmd:
    vx = 0.5
    vy = 0.0
    dyaw = 0.0


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        if key in config:
            config[key] = np.array(config[key], dtype=np.float32)

    config['PLOT'] = bool(config.get('PLOT', False))
    config['plot_decimation'] = int(config.get('plot_decimation', 1))
    return config


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_gravity_orientation(quat):
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return quat_rotate_inverse(quat, gravity_vec)


def compute_observation(d, config, action_train, cmd_obj, n_joints, phase):
    """
    观测保持你原来结构（做了少量清理，不打印）。
    """
    qj = d.qpos[7:7+n_joints].copy()
    qj = mujoco2train(qj)
    dqj = d.qvel[6:6+n_joints].copy()
    dqj = mujoco2train(dqj)

    quat = d.qpos[3:7].copy()  # [w,x,y,z]
    omega_world = d.qvel[3:6].copy()

    padded_defaults = config['default_angles'][:n_joints].copy()
    padded_defaults = mujoco2train(padded_defaults)

    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega_world * config['ang_vel_scale']

    cmd_vec = np.array([float(cmd_obj.vx), float(cmd_obj.vy), float(cmd_obj.dyaw)], dtype=np.float32)

    single_obs_dim = 3 + 3 + 3 + n_joints + n_joints + n_joints
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)

    single_obs[0:3] = cmd_vec * config['cmd_scale']
    single_obs[3:6] = omega_scaled
    single_obs[6:9] = gravity_orientation
    single_obs[9:9+n_joints] = qj_scaled
    single_obs[9+n_joints:9+2*n_joints] = dqj_scaled
    single_obs[9+2*n_joints:9+3*n_joints] = action_train

    return single_obs, single_obs_dim


def exp_avg_filter(x, avg, alpha=0.95):
    return alpha * x + (1 - alpha) * avg


# =========================
# 速度记录
# =========================
speed_data = {
    "time": [],
    "cmd_vx": [], "cmd_vy": [], "cmd_dyaw": [],
    "vx": [], "vy": [], "dyaw": [],
    "err_vx": [], "err_vy": [], "err_dyaw": []
}


def clear_speed_data():
    for k in speed_data.keys():
        speed_data[k].clear()


def save_speed_data(save_path):
    df = pd.DataFrame(speed_data)
    df["abs_err_vx"] = np.abs(df["err_vx"])
    df["abs_err_vy"] = np.abs(df["err_vy"])
    df["abs_err_dyaw"] = np.abs(df["err_dyaw"])
    df.to_csv(save_path, index=False)
    print(f"[ok] speed tracking csv saved: {save_path}")
    return df


# =========================
# Main
# =========================
def main():
    config_path = Path(__file__).resolve().parent / 'configs' / 'go2.yaml'
    config = load_config(str(config_path))

    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = float(config['simulation_dt'])

    n_joints = int(config['num_actions'])

    # Policy
    ort_session = ort.InferenceSession(config['policy_path'], providers=['CPUExecutionProvider'])
    ort_input_name = ort_session.get_inputs()[0].name

    # buffers
    action_train = np.zeros(n_joints, dtype=np.float32)   # 训练顺序（policy输出）
    action_mj = np.zeros(n_joints, dtype=np.float32)      # Mujoco顺序（用于控制）
    action_flt = np.zeros(n_joints, dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()

    # obs history
    single_obs, single_obs_dim = compute_observation(d, config, action_train, cmd, n_joints, phase=0.0)
    obs_history = collections.deque(maxlen=int(config['obs_history_len']))
    for _ in range(int(config['obs_history_len'])):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    obs = np.zeros(int(config['num_obs']), dtype=np.float32)

    # phase
    cycle_time = float(config.get('cycle_time', 0.6))
    phase_length_buf = 0

    keyboard_cmd_handler = KeyboardCmdHandler(step=0.1)

    # 渲染节流：尽量让viewer 50~60Hz
    render_stride = max(1, int(round(1.0 / (m.opt.timestep * max(1, 50)))))

    counter = 0
    start_wall = time.time()

    # csv name
    policy_stem = Path(config['policy_path']).stem
    csv_path = str(Path(CSV_SAVE_DIR) / f"{CSV_NAME_PREFIX}_{policy_stem}.csv")

    with mujoco.viewer.launch_passive(m, d, key_callback=keyboard_callback) as viewer:
        while viewer.is_running() and (time.time() - start_wall) < float(config['simulation_duration']):
            step_start = time.time()

            # reset
            if kb_cmd.reset:
                mujoco.mj_resetData(m, d)
                action_train[:] = 0
                action_mj[:] = 0
                action_flt[:] = 0
                target_dof_pos = config['default_angles'].copy()
                phase_length_buf = 0
                keyboard_cmd_handler.linear_x = 0
                keyboard_cmd_handler.linear_y = 0
                keyboard_cmd_handler.angular_z = 0

                obs_history.clear()
                for _ in range(int(config['obs_history_len'])):
                    obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
                obs[:] = 0

                clear_speed_data()

                kb_cmd.reset = False
                print("[info] reset done.")

            # --- low-level PD control using last action_mj ---
            action_scaled = action_mj * config['action_scales']
            action_flt = exp_avg_filter(action_scaled, action_flt, alpha=1.0)  # 你原来就是1
            target_dof_pos = action_flt + config['default_angles']

            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+n_joints],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+n_joints],
                config['kds']
            )
            d.ctrl[:n_joints] = leg_tau

            # step physics
            mujoco.mj_step(m, d)
            counter += 1

            # --- control update (policy inference) ---
            if counter % int(config['control_decimation']) == 0:
                # update cmd from keyboard
                keyboard_cmd_handler.update(kb_cmd)
                # cmd.vx = float(keyboard_cmd_handler.linear_x)
                cmd.vx = 3
                cmd.vy = float(keyboard_cmd_handler.linear_y)
                cmd.dyaw = float(keyboard_cmd_handler.angular_z)
                print(f"cmd_vx: {cmd.vx:.3f}, cmd_vy: {cmd.vy:.3f}, cmd_dyaw: {cmd.dyaw:.3f}")
                kb_cmd.zero()

                # ========== log speed tracking (body frame) ==========
                quat = d.qpos[3:7].copy()
                lin_vel_world = d.qvel[0:3].copy()
                ang_vel_world = d.qvel[3:6].copy()
                lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)
                ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)

                vx = float(lin_vel_body[0])
                vy = float(lin_vel_body[1])
                dyaw = float(ang_vel_body[2])

                speed_data["time"].append(float(d.time))
                speed_data["cmd_vx"].append(float(cmd.vx))
                speed_data["cmd_vy"].append(float(cmd.vy))
                speed_data["cmd_dyaw"].append(float(cmd.dyaw))
                speed_data["vx"].append(vx)
                speed_data["vy"].append(vy)
                speed_data["dyaw"].append(dyaw)
                speed_data["err_vx"].append(vx - float(cmd.vx))
                speed_data["err_vy"].append(vy - float(cmd.vy))
                speed_data["err_dyaw"].append(dyaw - float(cmd.dyaw))
                # ====================================================

                # phase
                phase_length_buf += 1
                phase = (phase_length_buf * float(config['simulation_dt']) * int(config['control_decimation']) / cycle_time) % 1.0

                # obs
                single_obs, _ = compute_observation(d, config, action_train, cmd, n_joints, phase)
                single_obs = np.clip(single_obs, -100, 100)
                obs_history.append(single_obs)

                for i, hist_obs in enumerate(obs_history):
                    s = i * single_obs_dim
                    e = s + single_obs_dim
                    obs[s:e] = hist_obs

                # policy inference
                input_batch = obs.astype(np.float32)[None, :]
                onnx_outputs = ort_session.run(None, {ort_input_name: input_batch})
                action_train = onnx_outputs[0].squeeze().astype(np.float32)
                action_mj = train2mujoco(action_train)

                # render
                if (counter % render_stride) == 0:
                    viewer.sync()

            # keep real-time (optional)
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # save csv
    if SAVE_CSV and len(speed_data["time"]) > 0:
        save_speed_data(csv_path)


if __name__ == "__main__":
    main()
