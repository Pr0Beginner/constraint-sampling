import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# ==========================================
# 全局学术图表规范设置 (CCF-A类 标准风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
# 增加全局默认字体基准大小（可选）
plt.rcParams['font.size'] = 14

# ==========================================
# 数据准备：实验基础路径配置
# ==========================================
# 这里只需填入文件的“基础前缀”（不包含后缀部分）
# 代码会自动补全 _best_history.xlsx, _ControlOnly_best_history.xlsx 等
base_experiments = [
    # r"E1_20",
    # r"E1_50",
    # r"E2_20",
    # r"E2_50",
    # r"g04_20",
    # r"g04_50",
    # r"g09_20",
    # r"g09_50",
    r"g01_20",
    r"g01_50"
]

# 名称映射字典：将 gxx 转换为 Ex
name_mapping = {
    "g04": "E3",
    "g09": "E4",
    "g01": "E5"
}


def generate_title(base_name_raw, index):
    """根据基础前缀生成学术标题，如 (a) E1 (N=20) Convergence"""
    label = f"({chr(97 + index)})"
    file_name = os.path.basename(base_name_raw)

    parts = file_name.split('_')
    base_name = parts[0]
    n_value = parts[1] if len(parts) > 1 else ""

    if base_name in name_mapping:
        base_name = name_mapping[base_name]

    if n_value.isdigit():
        return f"{label} {base_name} (N={n_value}) Convergence"
    return f"{label} {base_name} Convergence"


# ==========================================
# 画布排版设置 (每行固定 2 个图)
# ==========================================
num_files = len(base_experiments)
ncols = 2
nrows = math.ceil(num_files / ncols)

# 每行分配约 3.5 英寸的高度，整体更紧凑美观
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.5 * nrows))
axes = axes.flatten() if nrows > 1 else (np.array([axes]) if num_files == 1 else axes.flatten())

# ==========================================
# 绘图逻辑循环 (消融实验三线合一)
# ==========================================
for i, base_prefix in enumerate(base_experiments):
    ax = axes[i]

    # 构建三个文件的完整路径
    # 1. 完整模块: FGCS
    file_full = f"{base_prefix}_best_history.xlsx"
    # 2. 保留边界控制，去除排斥: FGCS w/o E-Rep
    file_ctrl = f"{base_prefix}_ControlOnly_best_history.xlsx"
    # 3. 保留排斥，去除边界控制: FGCS w/o CO
    file_rep = f"{base_prefix}_RepulsionOnly_best_history.xlsx"

    # 定义绘图配置 (为了学术规范，采用不同颜色、线型和粗细以区分主次)
    plot_configs = [
        {"path": file_full, "label": "FGCS", "color": "#d62728", "linestyle": "-", "linewidth": 2.0, "zorder": 3},
        # 红色实线，最粗，层级最高
        {"path": file_ctrl, "label": "FGCS w/o E-Rep", "color": "#1f77b4", "linestyle": "--", "linewidth": 1.5,
         "zorder": 2},  # 蓝色虚线
        {"path": file_rep, "label": "FGCS w/o CO", "color": "#2ca02c", "linestyle": "-.", "linewidth": 1.5, "zorder": 1}
        # 绿色点划线
    ]

    max_x_val = 0

    for config in plot_configs:
        try:
            if not os.path.exists(config["path"]):
                print(f"警告: 找不到文件 {config['path']}")
                continue

            df = pd.read_excel(config["path"])
            x = df.iloc[:, 0]  # 迭代次数
            y = df.iloc[:, 1]  # 均匀性准则值

            ax.plot(x, y, label=config["label"], color=config["color"],
                    linestyle=config["linestyle"], linewidth=config["linewidth"], zorder=config["zorder"])

            max_x_val = max(max_x_val, int(x.max()))
        except Exception as e:
            print(f"读取或处理文件 {config['path']} 失败: {e}")

    # X轴刻度设置 (间隔 300)
    if max_x_val > 0:
        xticks = np.arange(0, max_x_val + 1, 300)
        ax.set_xticks(xticks)

    # ==========================================
    # 坐标轴与刻度样式设置
    # ==========================================
    ax.tick_params(axis='both', which='major', labelsize=13, direction='in')
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Criterion Value', fontsize=15)
    ax.set_title(generate_title(base_prefix, i), fontsize=16, pad=10)

    # 网格与图例设置
    ax.grid(True, linestyle=':', color='gray', alpha=0.5)
    ax.legend(fontsize=13, loc='upper right', frameon=False)

# ==========================================
# 版面调整与保存
# ==========================================
# 隐藏多余的空白子图 (如果总数是奇数)
for j in range(num_files, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# 保存高清 PDF 与 PNG
output_pdf = 'ablation_study_convergence.pdf'
output_png = 'ablation_study_convergence.png'

# 【主要修改点】：将 dpi=300 修改为了 dpi=600，如果觉得还不够可以改成 1000 或 1200
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.savefig(output_png, dpi=600, bbox_inches='tight')

print(f"绘图完成！消融对比图已保存为 {output_pdf} 和 {output_png}")
plt.show()
