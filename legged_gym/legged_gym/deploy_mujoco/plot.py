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
    # "robot_pose_disturbance_data.csv", 
    "robot_pose_disturbance_data_mod8.csv",
    "robot_pose_disturbance_data_dream.csv",
    "robot_pose_disturbance_data_him.csv",
    # "robot_pose_disturbance_data_mod8.csv"
]

# 2. 每个文件对应的标签（用于图例显示）
FILE_LABELS = [
    "mod8",
    "dreamwaq",
    "himloco",
    # "mod8",
    
# "wo/attention",
#     "himloco",
# "wo/barlow loss",
    
    
]

# 3. 每个文件对应的颜色（可自定义，也可让脚本自动分配）
LINE_COLORS = [
    "#1f77b4",  # 蓝色
    "#ff7f0e",  # 橙色
    "#2ca02c",  # 绿色
    "#d62728",   # 红色（如需更多文件可追加）
    "#17becf",  # 青蓝
]

# 4. 扰动触发时间（用于图中标注，单位：秒）
DISTURBANCE_TRIGGER_TIME = 1.0

# 5. 输出图片路径和格式
OUTPUT_PITCH_PATH = "pitch_angle_comparison.png"  # 俯仰角对比图
OUTPUT_ROLL_PATH = "roll_angle_comparison.png"    # 横滚角对比图

# 6. 绘图样式配置
PLOT_CONFIG = {
    "figsize": (12, 6),       # 图片尺寸（宽, 高）
    "dpi": 400,               # 分辨率（越高越清晰）
    "font_size": 18,          # 基础字体大小
    "line_width": 2,          # 曲线宽度
    "grid": True,             # 是否显示网格
    "legend_loc": "lower right",  # 核心修改：图例位置改为右下角
    "x_label": "Time (s)",    # X轴标签
    "y_label_pitch": "Pitch Angle (°)",  # 俯仰角Y轴标签
    "y_label_roll": "Roll Angle (°)",    # 横滚角Y轴标签
    # "title_pitch": "Pitch Angle Comparison Under External Disturbance",  # 俯仰角标题
    # "title_roll": "Roll Angle Comparison Under External Disturbance",    # 横滚角标题
    "disturbance_line_style": "--",  # 扰动触发线样式
    "disturbance_line_color": "black",  # 扰动触发线颜色
    "disturbance_line_width": 1.5,      # 扰动触发线宽度
    "disturbance_label": "Disturbance Trigger (1s)",  # 扰动触发线标签
    "output_pitch_path": OUTPUT_PITCH_PATH,  # 新增：俯仰角图片保存路径
    "output_roll_path": OUTPUT_ROLL_PATH      # 新增：横滚角图片保存路径
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

def plot_angle_comparison(data_list, labels, colors, angle_type="pitch", config=PLOT_CONFIG):
    """
    绘制角度对比图
    :param data_list: 加载的DataFrame列表
    :param labels: 文件标签列表
    :param colors: 颜色列表
    :param angle_type: 角度类型（"pitch" 或 "roll"）
    :param config: 绘图配置
    """
    # 初始化画布
    plt.figure(figsize=config["figsize"], dpi=config["dpi"])
    
    # 设置全局字体大小
    plt.rcParams.update({"font.size": config["font_size"]})
    
    # 绘制每个文件的曲线
    for i, (df, label) in enumerate(zip(data_list, labels)):
        color = colors[i] if i < len(colors) else None
        if angle_type == "pitch":
            y_data = df["pitch_deg"]
            y_label = config["y_label_pitch"]
            # title = config["title_pitch"]
        else:
            y_data = df["roll_deg"]
            y_label = config["y_label_roll"]
            # title = config["title_roll"]
        
        plt.plot(
            df["time"], y_data,
            label=label,
            color=color,
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
    plt.ylabel(y_label)
    # plt.title(title)
    
    # 显示网格和图例
    if config["grid"]:
        plt.grid(True, alpha=0.3)
    plt.legend(loc=config["legend_loc"])
    
    # 调整布局（防止标签被截断）
    plt.tight_layout()
    
    # 修复：正确获取保存路径
    save_path = config["output_pitch_path"] if angle_type == "pitch" else config["output_roll_path"]
    plt.savefig(save_path, bbox_inches="tight")
    print(f"{angle_type.capitalize()}角对比图已保存到：{save_path}")
    
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
    valid_colors = LINE_COLORS[:len(data_list)]
    
    # 3. 绘制俯仰角对比图
    print("绘制俯仰角对比图...")
    plot_angle_comparison(data_list, valid_labels, valid_colors, angle_type="pitch")
    
    # 4. 绘制横滚角对比图
    print("绘制横滚角对比图...")
    plot_angle_comparison(data_list, valid_labels, valid_colors, angle_type="roll")
    
    print("所有图片绘制完成！")

if __name__ == "__main__":
    main()
