import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.lines as mlines

# ==========================================
# 1. 初始化画布和基础设置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.set_facecolor('#FFFFFF')

# 设置网格范围 (0 到 0.5)
x = np.linspace(0, 0.5, 1000)
y = np.linspace(0, 0.5, 1000)
X, Y = np.meshgrid(x, y)

# ==========================================
# 2. 绘制约束条件与可行域
# ==========================================
# 约束不等式
C1 = Y <= X
C2 = Y >= X**2
C3 = (X - 1)**2 + (Y - 1)**2 >= 1
Feasible = C1 & C2 & C3

# 填充可行域 (浅绿色)
ax.contourf(X, Y, Feasible, levels=[0.5, 1.5], colors=['#E2F0D9'], alpha=0.9)

# 绘制约束边界线
ax.plot(x, x, color='m', linestyle='--', linewidth=1.5, zorder=2)             # y = x
ax.plot(x, x**2, color='g', linestyle='-', linewidth=1.5, zorder=2)            # y = x^2
circle = Circle((1, 1), 1, color='k', fill=False, linestyle='-', linewidth=1.5, zorder=2) # (x-1)^2 + (y-1)^2 = 1
ax.add_patch(circle)

# ==========================================
# 3. 绘制样本点
# ==========================================
samples = np.array([
    [0.18, 0.098], [0.22, 0.08], [0.23, 0.198], [0.225, 0.18],
    [0.265, 0.14], [0.285, 0.16], [0.36, 0.195], [0.35, 0.12]
])
ax.scatter(samples[:, 0], samples[:, 1], color='red', s=50, zorder=4)

# 目标精英点 Target Elite
target_elite = np.array([0.38, 0.18])
ax.scatter(*target_elite, color='orange', marker='*', s=250, zorder=5)

# ==========================================
# 4. 辅助函数：绘制精美箭头
# ==========================================
def draw_arrow(ax, start, vec, color, linestyle='-', linewidth=2, zorder=6):
    arrow = FancyArrowPatch(start, start + vec,
                            arrowstyle='-|>',
                            mutation_scale=15,
                            color=color,
                            linestyle=linestyle,
                            linewidth=linewidth,
                            zorder=zorder)
    ax.add_patch(arrow)

# ==========================================
# 5. 场景一：切线投影 (Tangent Projection)
# ==========================================
pt1 = np.array([0.22, 0.22])
ax.scatter(*pt1, color='blue', s=70, zorder=5)

d_desired = np.array([-0.04, 0.08])
d_tangent = np.array([0.025, 0.025])

ax.plot([pt1[0]-0.06, pt1[0]+0.06], [pt1[1]-0.06, pt1[1]+0.06], color='darkorange', linestyle='-.', linewidth=2, zorder=3)

draw_arrow(ax, pt1, d_desired, 'dodgerblue', linestyle='--', linewidth=2)
draw_arrow(ax, pt1, d_tangent, 'purple', linestyle='-', linewidth=3, zorder=7)

ax.plot([pt1[0]+d_desired[0], pt1[0]+d_tangent[0]],
        [pt1[1]+d_desired[1], pt1[1]+d_tangent[1]], color='gray', linestyle=':', linewidth=1.5, zorder=3)

# 场景一标注：微调避免重叠
ax.text(pt1[0]-0.045, pt1[1]+0.085, '$d_{期望}$', color='dodgerblue', fontsize=14, fontweight='bold')
ax.text(pt1[0]+0.025, pt1[1]+0.045, '$d_{切线}$ (滑动)', color='purple', fontsize=14, fontweight='bold')
ax.text(pt1[0]-0.07, pt1[1]-0.04, '切线\n空间', color='darkorange', fontsize=13, ha='center')

# ==========================================
# 6. 场景二：梯度修复 (Gradient Repair)
# ==========================================
pt2 = np.array([0.30, 0.03])
ax.scatter(*pt2, color='blue', s=70, zorder=5)

f_search = np.array([0.06, 0.10])
f_repair = np.array([-0.04, 0.08])
d_blended = np.array([-0.01, 0.095])

draw_arrow(ax, pt2, f_search, 'dodgerblue', linestyle='--', linewidth=2)
draw_arrow(ax, pt2, f_repair, 'darkorange', linestyle='-', linewidth=2)
draw_arrow(ax, pt2, d_blended, 'purple', linestyle='-', linewidth=3.5, zorder=7)

ax.plot([pt2[0]+f_search[0], pt2[0]+d_blended[0]],
        [pt2[1]+f_search[1], pt2[1]+d_blended[1]], color='gray', linestyle=':', linewidth=1.5, zorder=3)
ax.plot([pt2[0]+f_repair[0], pt2[0]+d_blended[0]],
        [pt2[1]+f_repair[1], pt2[1]+d_blended[1]], color='gray', linestyle=':', linewidth=1.5, zorder=3)

# 场景二标注
ax.text(pt2[0]+0.065, pt2[1]+0.13, '$F_{搜索}$', color='dodgerblue', fontsize=14, fontweight='bold')
ax.text(pt2[0]-0.085, pt2[1]+0.06, '$F_{修复}$', color='darkorange', fontsize=14, fontweight='bold')
ax.text(pt2[0]-0.08, pt2[1]+0.12, '$d_{混合}$ (修复)', color='purple', fontsize=14, fontweight='bold')

# ==========================================
# 7. 全局标注与排版美化
# ==========================================
# 可行域/不可行域标注
ax.text(0.38, 0.32, '可行域', color='forestgreen', fontsize=18, ha='center')
ax.text(0.44, 0.07, '不可行域', color='gray', fontsize=18, ha='center')

# 坐标轴与标题
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)
ax.set_xlabel('因子 x', fontsize=18)
ax.set_ylabel('因子 y', fontsize=18)
ax.set_title('约束边界控制：切线投影与梯度修复', fontsize=20, pad=15)

ax.tick_params(axis='both', which='major', labelsize=14)

# 图例
legend_elements = [
    mlines.Line2D([], [], color='m', linestyle='--', label='y = x'),
    mlines.Line2D([], [], color='g', linestyle='-', label='y = x²'),
    mlines.Line2D([], [], color='k', linestyle='-', label='$(x-1)^2 + (y-1)^2 = 1$'),
    mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=7, label='样本点'),
    mlines.Line2D([], [], color='orange', marker='*', linestyle='None', markersize=14, label='目标精英点'),
    mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=7, label='当前点 ($x_i$)'),
    mlines.Line2D([], [], color='dodgerblue', linestyle='--', linewidth=2.5, label='$d_{期望}$ / $F_{搜索}$ (探索)'),
    mlines.Line2D([], [], color='darkorange', linestyle='-', linewidth=2.5, label='切线空间 / $F_{修复}$'),
    mlines.Line2D([], [], color='purple', linestyle='-', linewidth=4, label='$d_{切线}$ / $d_{混合}$ (最终方向)')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=True, edgecolor='black', facecolor='white', framealpha=0.95, fontsize=11)

ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
