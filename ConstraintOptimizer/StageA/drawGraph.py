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

# ==========================================
# 数据准备
# ==========================================
# 注意：包含 \ 的 Windows 路径前加上了 r，防止转义字符报错
file_paths = [
    # "E1_20_best_history.xlsx",
    # "E1_50_best_history.xlsx",
    # "E2_20_best_history.xlsx",
    # "E2_50_best_history.xlsx",
    # r"D:\PythonProjects\CMS\local\constrained-sampling\ConstraintOptimizer\g04_20_best_history.xlsx",
    # r"D:\PythonProjects\CMS\local\constrained-sampling\ConstraintOptimizer\g04_50_best_history.xlsx",
    # r"D:\PythonProjects\CMS\local\constrained-sampling\ConstraintOptimizer\g09_20_best_history.xlsx",
    # r"D:\PythonProjects\CMS\local\constrained-sampling\ConstraintOptimizer\g09_50_best_history.xlsx",
    r"D:\PythonProjects\CMS\local\constrained-sampling\ConstraintOptimizer\g01_20_best_history.xlsx",
    r"D:\PythonProjects\CMS\local\constrained-sampling\ConstraintOptimizer\g01_50_best_history.xlsx",
]

# 名称映射字典：将 gxx 转换为 Ex
name_mapping = {
    "g04": "E3",
    "g09": "E4",
    "g01": "E5"
}

def generate_title(file_path, index):
    """根据文件路径自动提取信息并生成学术标题"""
    # 1. 生成 (a), (b), (c)...
    label = f"({chr(97 + index)})"

    # 2. 提取纯文件名（不含路径）
    file_name = os.path.basename(file_path)

    # 3. 解析文件名获取实验名称和样本数 (如 g04 和 20)
    # 假设命名规则都是 "名称_样本数_xxx.xlsx"
    parts = file_name.split('_')
    base_name = parts[0]
    n_value = parts[1] if len(parts) > 1 else ""

    # 4. 执行映射
    if base_name in name_mapping:
        base_name = name_mapping[base_name]

    # 5. 组合标题
    if n_value.isdigit():
        return f"{label} {base_name} (N={n_value}) Convergence"
    else:
        return f"{label} {base_name} Convergence"

# ==========================================
# 动态计算画布排版 (每行固定 2 个图)
# ==========================================
num_files = len(file_paths)
ncols = 2
nrows = math.ceil(num_files / ncols)  # 向上取整，10个文件就是5行

# 动态设置高度：每行分配约 4.5 英寸的高度，为大字体留出足够空间
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4.5 * nrows))

# 将多维轴数组展平，方便使用一维索引循环
if nrows == 1:
    axes = np.array([axes]) if num_files == 1 else axes.flatten()
else:
    axes = axes.flatten()

# ==========================================
# 绘图逻辑循环
# ==========================================
for i, file_path in enumerate(file_paths):
    ax = axes[i]
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        x = df.iloc[:, 0]  # 第一列：迭代次数
        y = df.iloc[:, 1]  # 第二列：均匀性准则值

        # 绘制折线 (将线条加粗为2.0，使视觉更清晰)
        ax.plot(x, y, color='#1f77b4', linewidth=2.0, label='Criterion Value')

        # 动态设置横坐标间隔为 300
        max_x = int(x.max())
        xticks = np.arange(0, max_x + 1, 300)
        ax.set_xticks(xticks)

        # 坐标轴刻度样式：向内(in)，大幅放大刻度字体至 14
        ax.tick_params(axis='both', which='major', labelsize=14, direction='in')

        # 设置坐标轴标签，大幅放大标签字体至 16
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('Criterion Value', fontsize=16)

        # 调用我们自己写的动态标题生成函数，大幅放大标题字体至 18，并增加 pad
        title_str = generate_title(file_path, i)
        ax.set_title(title_str, fontsize=18, pad=12)

        # 网格与图例，放大图例字体至 14
        ax.grid(True, linestyle='--', color='gray', alpha=0.3)
        ax.legend(fontsize=14, loc='upper right', frameon=False)

    except Exception as e:
        print(f"读取或处理文件 {file_path} 失败: {e}")
        # 如果当前图绘制失败，依然保留图框或者可以清空该轴
        ax.set_title(f"({chr(97 + i)}) Error loading data", fontsize=16)

# ==========================================
# 版面调整与保存
# ==========================================
# 若文件数是奇数，将最后一个多余的空白子图隐藏掉
for j in range(num_files, len(axes)):
    fig.delaxes(axes[j])

# 自适应调整子图间距，增加 pad 避免文字过大互相堆叠
plt.tight_layout(pad=2.0)

# 导出为科研规范的 PDF 以及高分辨率 PNG，增加 SVG 矢量格式
output_pdf = 'all_convergence_curves.pdf'
output_svg = 'all_convergence_curves.svg'
output_png = 'all_convergence_curves.png'

# pdf 和 svg 是天然的无损矢量图，投稿首选
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.savefig(output_svg, format='svg', bbox_inches='tight')
plt.savefig(output_png, dpi=600, bbox_inches='tight')

print(f"绘图完成！\n矢量图已保存为: {output_pdf} 和 {output_svg}\n高分辨率图已保存为: {output_png}")
plt.show()
