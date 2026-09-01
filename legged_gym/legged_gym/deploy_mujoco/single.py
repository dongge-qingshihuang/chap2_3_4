import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ====================== 配置项（根据你的需求修改）======================
# 1. 待读取的CSV文件路径列表（按需要对比的顺序排列）
CSV_FILES = [
    # "robot_pose_disturbance_data_bar_150N.csv",  # 第一个测试案例
    # "robot_pose_disturbance_data_dream.csv",  # 第二个测试案例
    # "robot_pose_disturbance_data_wobarlow.csv",   # 第三个测试案例（可增减）
    # "robot_pose_disturbance_data_him.csv", 
    "robot_pose_disturbance_data_bar_240N.csv", 
]

# 2. 每个文件对应的标签（用于图例显示）
FILE_LABELS = [
    "ours",
#     "dreamwaq",
# "wo/dha",
    # "him",
# "wo/barlow loss",
]

# 3. 颜色配置：为每个文件的pitch和roll分别指定颜色
# 格式：[(pitch_color1, roll_color1), (pitch_color2, roll_color2), ...]
LINE_COLOR_PAIRS = [
    ("#1f77b4", "#ff7f0e"),  # ours: pitch蓝色, roll橙色
    ("#2ca02c", "#d62728"),  # 第二个文件: pitch绿色, roll红色（如需追加）
    ("#17becf", "#9467bd"),  # 第三个文件: pitch青蓝, roll紫色（如需追加）
]

# 4. 扰动触发时间（用于图中标注，单位：秒）
DISTURBANCE_TRIGGER_TIME = 1.0

# 5. 输出图片路径和格式
OUTPUT_COMBINED_PATH = "pose_angle_comparison.png"  # 合并后的姿态角对比图

# 6. 绘图样式配置
PLOT_CONFIG = {
    "figsize": (12, 6),       # 图片尺寸（宽, 高）
    "dpi": 300,               # 分辨率（越高越清晰）
    "font_size": 18,          # 基础字体大小
    "line_width": 2,          # 曲线宽度
    "grid": True,             # 是否显示网格
    "legend_loc": "lower right",  # 核心修改：图例位置改为右下角
    "x_label": "Time (s)",    # X轴标签
    "y_label": "Angle (°)",   # 合并后的Y轴标签（包含pitch和roll）
    # "title": "Pitch & Roll Angle Comparison Under External Disturbance",  # 合并后的标题
    "disturbance_line_style": "--",  # 扰动触发线样式
    "disturbance_line_color": "black",  # 扰动触发线颜色
    "disturbance_line_width": 1.5,      # 扰动触发线宽度
    "disturbance_label": "Disturbance Trigger (1s)",  # 扰动触发线标签
    "output_combined_path": OUTPUT_COMBINED_PATH,  # 合并图保存路径
    # 角度标签后缀
    "pitch_suffix": " (Pitch)",# pitch标签后缀
    "roll_suffix": " (Roll)"   # roll标签后缀
}
# ======================================================================

def load_pose_data(file_paths):
    """加载多个CSV文件的姿态数据"""
    data_list = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"警告：文件 {file_path} 不存在，跳过！")
            continue
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 检查必要列是否存在
        required_cols = ["time", "pitch_deg", "roll_deg"]
        if not all(col in df.columns for col in required_cols):
            print(f"警告：文件 {file_path} 缺少必要列（{required_cols}），跳过！")
            continue
        # 清理数据（去除空值、异常值）
        df = df.dropna(subset=required_cols)
        df = df[(np.abs(df["pitch_deg"]) < 90) & (np.abs(df["roll_deg"]) < 90)]  # 过滤异常角度
        data_list.append(df)
    return data_list

def plot_combined_angle_comparison(data_list, labels, color_pairs, config=PLOT_CONFIG):
    """
    将pitch和roll绘制在同一张图中（不同颜色）
    :param data_list: 加载的DataFrame列表
    :param labels: 文件标签列表
    :param color_pairs: 每个文件的pitch/roll颜色对列表
    :param config: 绘图配置
    """
    # 初始化画布
    plt.figure(figsize=config["figsize"], dpi=config["dpi"])
    
    # 设置全局字体大小
    plt.rcParams.update({"font.size": config["font_size"]})
    
    # 绘制每个文件的pitch和roll曲线（不同颜色）
    for i, (df, label) in enumerate(zip(data_list, labels)):
        # 获取当前文件的pitch/roll颜色对
        pitch_color, roll_color = color_pairs[i] if i < len(color_pairs) else ("#1f77b4", "#ff7f0e")
        
        # 绘制俯仰角
        plt.plot(
            df["time"], df["pitch_deg"],
            label=f"{label}{config['pitch_suffix']}",
            color=pitch_color,
            linewidth=config["line_width"]
        )
        
        # 绘制横滚角（不同颜色）
        plt.plot(
            df["time"], df["roll_deg"],
            label=f"{label}{config['roll_suffix']}",
            color=roll_color,
            linewidth=config["line_width"]
        )
    
    # 绘制扰动触发线
    plt.axvline(
        x=DISTURBANCE_TRIGGER_TIME,
        linestyle=config["disturbance_line_style"],
        color=config["disturbance_line_color"],
        linewidth=config["disturbance_line_width"],
        label=config["disturbance_label"]
    )
    
    # 设置轴标签和标题
    plt.xlabel(config["x_label"])
    plt.ylabel(config["y_label"])
    # plt.title(config["title"])
    
    # 显示网格和图例
    if config["grid"]:
        plt.grid(True, alpha=0.3)
    plt.legend(loc=config["legend_loc"])  # 使用配置中的图例位置
    
    # 调整布局（防止标签被截断）
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(config["output_combined_path"], bbox_inches="tight")
    print(f"合并后的姿态角对比图已保存到：{config['output_combined_path']}")
    
    # 显示图片（可选）
    plt.show()

def main():
    # 1. 加载数据
    print("开始加载CSV文件...")
    data_list = load_pose_data(CSV_FILES)
    if len(data_list) == 0:
        print("错误：未加载到任何有效数据，程序退出！")
        return
    
    # 2. 裁剪标签和颜色列表（匹配有效数据数量）
    valid_labels = FILE_LABELS[:len(data_list)]
    valid_color_pairs = LINE_COLOR_PAIRS[:len(data_list)]
    
    # 3. 绘制合并后的姿态角对比图（pitch+roll，不同颜色）
    print("绘制合并后的俯仰角&横滚角对比图...")
    plot_combined_angle_comparison(data_list, valid_labels, valid_color_pairs)
    
    print("所有图片绘制完成！")

if __name__ == "__main__":
    main()
