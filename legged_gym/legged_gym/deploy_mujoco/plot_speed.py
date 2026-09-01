import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 你只需要改这里
# =========================
CSV_FILES = [
    ("speed_tracking_mod8_3.csv", "ours"),
    ("speed_tracking_dream_3.csv", "dreamwaq"),
]

PLOT_COMPONENTS = ["vx", "vy", "dyaw"]

TIME_RANGE = None        # 例如 (0.0, 12.0)
ZERO_TIME = True

EMA_ALPHA = None         # 例如 0.9；None不平滑

OUT_DIR = "plots_speed_tracking"

# cmd一致性检查阈值（最大绝对差超过就报警）
CMD_CHECK_ATOL = 1e-3
# =========================


def ema_filter(x, alpha):
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
    return y


def load_and_slice(csv_path, time_range=None, zero_time=True):
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 time 列")

    if zero_time and len(df) > 0:
        df["time"] = df["time"] - df["time"].iloc[0]

    if time_range is not None:
        t0, t1 = time_range
        mask = np.ones(len(df), dtype=bool)
        if t0 is not None:
            mask &= (df["time"] >= t0)
        if t1 is not None:
            mask &= (df["time"] < t1)
        df = df.loc[mask].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"{csv_path} 在筛选区间内无数据")
    return df


def compute_metrics(err):
    err = np.asarray(err, dtype=float)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    maxae = float(np.max(np.abs(err)))
    return rmse, mae, maxae


def align_by_time(df_ref, df_other, cols):
    """
    用最简单可靠的方式对齐：按最短长度裁剪（前提：你每个策略记录频率一致）。
    如果你两个CSV采样频率/长度不一致，建议改成按时间插值对齐，我也可以给你加。
    """
    n = min(len(df_ref), len(df_other))
    out_ref = df_ref.iloc[:n].copy()
    out_other = df_other.iloc[:n].copy()
    # time也裁剪
    return out_ref, out_other


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 先读所有csv
    dfs = []
    for csv_path, label in CSV_FILES:
        df = load_and_slice(csv_path, TIME_RANGE, ZERO_TIME)
        dfs.append((df, label, csv_path))

    # 选第一个作为“cmd参考”（因为你说期望值每个策略都一样）
    df_cmd_ref, ref_label, ref_path = dfs[0]

    # ========= 1) cmd一致性检查 =========
    for comp in PLOT_COMPONENTS:
        cmd_col = f"cmd_{comp}"
        if cmd_col not in df_cmd_ref.columns:
            raise ValueError(f"{ref_path} 缺少 {cmd_col}")

        for df_i, label_i, path_i in dfs[1:]:
            if cmd_col not in df_i.columns:
                raise ValueError(f"{path_i} 缺少 {cmd_col}")

            a, b = align_by_time(df_cmd_ref, df_i, [cmd_col])
            diff = (a[cmd_col].to_numpy() - b[cmd_col].to_numpy())
            max_abs = float(np.max(np.abs(diff)))
            if max_abs > CMD_CHECK_ATOL:
                print(f"[WARN] cmd_{comp} 在 {ref_label} vs {label_i} 不一致: max|diff|={max_abs:.6f}  (>{CMD_CHECK_ATOL})")
            else:
                print(f"[OK] cmd_{comp} 一致性通过: {ref_label} vs {label_i}, max|diff|={max_abs:.6f}")

    # ========= 2) 画 cmd vs actual =========
    for comp in PLOT_COMPONENTS:
        cmd_col = f"cmd_{comp}"
        act_col = comp

        plt.figure()

        # cmd 只画一次（用参考策略的cmd）
        t_cmd = df_cmd_ref["time"].to_numpy(dtype=float)
        cmd_vals = df_cmd_ref[cmd_col].to_numpy(dtype=float)
        if EMA_ALPHA is not None:
            cmd_plot = ema_filter(cmd_vals, EMA_ALPHA)
        else:
            cmd_plot = cmd_vals

        plt.plot(t_cmd, cmd_plot, label=f"cmd ({cmd_col})", linewidth=2)

        # 各策略 actual 都画
        for df_i, label_i, path_i in dfs:
            if act_col not in df_i.columns:
                raise ValueError(f"{path_i} 缺少 {act_col}")

            # 为了和cmd同轴，裁到共同长度
            a, b = align_by_time(df_cmd_ref, df_i, [act_col])
            t = a["time"].to_numpy(dtype=float)
            act_vals = b[act_col].to_numpy(dtype=float)

            if EMA_ALPHA is not None:
                act_plot = ema_filter(act_vals, EMA_ALPHA)
            else:
                act_plot = act_vals

            plt.plot(t, act_plot, label=f"actual ({label_i})")

        plt.xlabel("Time (s)")
        plt.ylabel(comp)
        plt.title(f"{comp}: cmd vs actual")
        plt.grid(True)
        plt.legend()

        out_path = os.path.join(OUT_DIR, f"cmd_vs_actual_{comp}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[ok] saved: {out_path}")

    # ========= 3) 画 error 曲线 =========
    for comp in PLOT_COMPONENTS:
        cmd_col = f"cmd_{comp}"
        act_col = comp
        err_col = f"err_{comp}"

        plt.figure()

        # 参考cmd（用于现算误差，保证所有策略用同一cmd）
        t_cmd = df_cmd_ref["time"].to_numpy(dtype=float)
        cmd_vals_ref = df_cmd_ref[cmd_col].to_numpy(dtype=float)

        for df_i, label_i, path_i in dfs:
            # 对齐长度
            a, b = align_by_time(df_cmd_ref, df_i, [act_col])
            t = a["time"].to_numpy(dtype=float)
            cmd_vals = a[cmd_col].to_numpy(dtype=float)  # 同一参考cmd（被裁剪）
            if err_col in b.columns:
                err = b[err_col].to_numpy(dtype=float)
            else:
                if act_col not in b.columns:
                    raise ValueError(f"{path_i} 缺少 {act_col}")
                err = (b[act_col].to_numpy(dtype=float) - cmd_vals)

            if EMA_ALPHA is not None:
                err_plot = ema_filter(err, EMA_ALPHA)
            else:
                err_plot = err

            rmse, mae, maxae = compute_metrics(err)
            print(f"[{label_i}] {comp}: RMSE={rmse:.4f}, MAE={mae:.4f}, MaxAE={maxae:.4f}")

            plt.plot(t, err_plot, label=label_i)

        plt.axhline(y=0.0, linestyle=":")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Error ({comp})  [actual - cmd]")
        plt.title(f"Speed Tracking Error: {comp}")
        plt.grid(True)
        plt.legend()

        out_path = os.path.join(OUT_DIR, f"error_{comp}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[ok] saved: {out_path}")

    # ========= 4) 线速度误差范数 sqrt(err_vx^2 + err_vy^2) =========
    plt.figure()
    for df_i, label_i, path_i in dfs:
        a, b = align_by_time(df_cmd_ref, df_i, ["vx", "vy"])
        t = a["time"].to_numpy(dtype=float)

        cmd_vx = a["cmd_vx"].to_numpy(dtype=float)
        cmd_vy = a["cmd_vy"].to_numpy(dtype=float)

        vx = b["vx"].to_numpy(dtype=float)
        vy = b["vy"].to_numpy(dtype=float)

        err_vx = vx - cmd_vx
        err_vy = vy - cmd_vy
        err_norm = np.sqrt(err_vx**2 + err_vy**2)

        if EMA_ALPHA is not None:
            err_norm_plot = ema_filter(err_norm, EMA_ALPHA)
        else:
            err_norm_plot = err_norm

        rmse, mae, maxae = compute_metrics(err_norm)
        print(f"[{label_i}] vel_norm: RMSE={rmse:.4f}, MAE={mae:.4f}, MaxAE={maxae:.4f}")

        plt.plot(t, err_norm_plot, label=label_i)

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Error Norm")
    plt.title("Speed Tracking Error Norm")
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(OUT_DIR, "error_vel_norm.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
