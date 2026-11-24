import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 使用更美观的图表样式
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.set_facecolor('#FFFFFF')

# 定义网格
x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)

# 约束条件
C1 = Y <= X
C2 = Y >= X ** 2
C3 = (X - 1) ** 2 + (Y - 1) ** 2 >= 1
Feasible = C1 & C2 & C3

# 绘制可行域和边界
ax.contourf(X, Y, Feasible, levels=[0.5, 1.5], colors=['#A5D6A7'], alpha=0.6, zorder=1)
ax.plot(x, x, color='#34495E', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
ax.plot(x, x ** 2, color='#34495E', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
circle_bound = Circle((1, 1), 1, color='#34495E', fill=False, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
ax.add_patch(circle_bound)

# 样本点
points = np.array([
    [0.25, 0.10], [0.30, 0.15], [0.20, 0.05], [0.35, 0.16], [0.15, 0.13],
    [0.40, 0.17], [0.10, 0.08], [0.28, 0.09], [0.22, 0.17], [0.32, 0.12]
])
ax.scatter(points[:, 0], points[:, 1], color='#3498DB', s=70, edgecolor='white', linewidth=1, label='Population Points', zorder=5)

# 指定点
current_point = points[0]
target_point = points[5]
neighbors = np.array([points[2], points[7], points[8]])

ax.scatter(*current_point, color='#F39C12', s=160, edgecolor='black', linewidth=1.5, zorder=6, label='Current Point $x_i$')
ax.scatter(*target_point, color='#9B59B6', s=200, marker='*', edgecolor='black', linewidth=1, zorder=6, label='Target Vector Direction')
ax.scatter(neighbors[:, 0], neighbors[:, 1], color='#E74C3C', s=90, edgecolor='white', linewidth=1, zorder=5, label='Local Neighbors')

# === 矢量合成核心逻辑 ===
alpha = 0.5
base_scale = 0.12  # 控制整体箭头基础长度

# 1. 灰色矢量 (Original Direction)
original_dir = target_point - current_point
original_dir_normed = original_dir / np.linalg.norm(original_dir)
vis_original = (1 - alpha) * original_dir_normed * base_scale

# 2. 计算总排斥力
repulsion_vec = np.zeros(2)
for n in neighbors:
    diff = current_point - n
    dist = np.linalg.norm(diff)
    dir_away = diff / dist
    weight = 1.0 / (dist ** 3)
    repulsion_vec += dir_away * weight
    # 画一条淡淡的虚线连接邻居和当前点，明确力的空间来源
    ax.plot([n[0], current_point[0]], [n[1], current_point[1]], color='#E74C3C', linestyle=':', linewidth=1.5, alpha=0.4)

repulsion_vec_normed = repulsion_vec / np.linalg.norm(repulsion_vec)
vis_repulsion = alpha * repulsion_vec_normed * base_scale

# 3. 绘制总红色排斥力和灰色原始方向
ax.quiver(*current_point, *vis_repulsion, angles='xy', scale_units='xy', scale=1,
          color='#E74C3C', width=0.006, zorder=6, label='Total Repulsion ($F_{repulsion}$)')
ax.quiver(*current_point, *vis_original, angles='xy', scale_units='xy', scale=1,
          color='#7F8C8D', width=0.006, zorder=5, label='Original Direction (Scaled)')

# 4. 灰色虚线延伸至 target_point
ax.plot([current_point[0] + vis_original[0], target_point[0]],
        [current_point[1] + vis_original[1], target_point[1]],
        color='#7F8C8D', linestyle=':', linewidth=1.5, alpha=0.6, label='Direction to Target')

# 5. 平行四边形法则展示 (黑 = 红 + 灰)
vis_combined = vis_original + vis_repulsion
# 红色末端到黑色末端 (平行于灰色)
ax.plot([current_point[0] + vis_repulsion[0], current_point[0] + vis_combined[0]],
        [current_point[1] + vis_repulsion[1], current_point[1] + vis_combined[1]],
        color='#7F8C8D', linestyle='--', linewidth=1.5, alpha=0.7)
# 灰色末端到黑色末端 (平行于红色)
ax.plot([current_point[0] + vis_original[0], current_point[0] + vis_combined[0]],
        [current_point[1] + vis_original[1], current_point[1] + vis_combined[1]],
        color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)

# 6. 黑色组合方向
ax.quiver(*current_point, *vis_combined, angles='xy', scale_units='xy', scale=1,
          color='#2C3E50', width=0.007, zorder=7, label='Combined Direction ($F_{combined}$)')

# 坐标轴与排版 (增大字体)
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.6)
ax.set_xlabel('Factor x', fontsize=18, fontweight='bold', color='#333333')
ax.set_ylabel('Factor y', fontsize=18, fontweight='bold', color='#333333')
ax.set_title('Evolutionary Direction Guidance Strategy\n(Vector Synthesis Details)',
             fontsize=20, fontweight='bold', color='#2C3E50', pad=20)

# 增大刻度字体
ax.tick_params(axis='both', which='major', labelsize=18)

legend = ax.legend(loc='upper right', fontsize=13, frameon=True, fancybox=True, framealpha=0.95, borderpad=1)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#CCCCCC')

ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.7, zorder=0)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
