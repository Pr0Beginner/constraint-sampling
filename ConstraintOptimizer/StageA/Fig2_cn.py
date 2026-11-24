'''
File name: 
Author: Zhao Mingyu
Date: 2026-07-10 12:47:50
Description: 
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 使用美观的图表样式
plt.style.use('seaborn-v0_8-ticks')  # 清爽的刻度线风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 仅保留一个面板 (a)，去掉 (b)，调整画布比例
fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

# ==========================================
# 空间分布与参考向量 (空间视图)
# ==========================================
ax1.set_title('参考向量的自适应更新（代际反馈）', pad=15)

r_max = 1.0
# 绘制背景边界
ax1.plot(r_max * np.cos(np.linspace(0, np.pi / 2, 100)), r_max * np.sin(np.linspace(0, np.pi / 2, 100)),
         color='#2C3E50', linestyle='-', linewidth=1.5, alpha=0.8)
ax1.plot([0, r_max], [0, 0], color='#2C3E50', linewidth=1.5, alpha=0.8)
ax1.plot([0, 0], [0, r_max], color='#2C3E50', linewidth=1.5, alpha=0.8)

# 划分 5 个参考向量 (对应 v1 到 v5)
angles_deg = [9, 27, 45, 63, 81]
vec_labels = ['$v_1$', '$v_2$', '$v_3$', '$v_4$', '$v_5$']
# 模拟不同向量关联的种群数量 n_k
# v1: 密集, v2: 中等, v3: 中等, v4: 极度稀疏, v5: 稀疏
n_k = [28, 12, 8, 0, 4]
weights = [0.05, 0.15, 0.20, 0.40, 0.20]  # 模拟更新后的权重 w_k

colors = ['#3498DB', '#95A5A6', '#95A5A6', '#E74C3C', '#E67E22']

np.random.seed(42)
for i, (angle, count, w, color, label) in enumerate(zip(angles_deg, n_k, weights, colors, vec_labels)):
    rad = np.radians(angle)
    dir_x, dir_y = np.cos(rad), np.sin(rad)

    # 绘制扇区辅助线
    ax1.plot([0, r_max * 0.95], [0, r_max * 0.95 * np.tan(rad)], color='#BDC3C7', linestyle=':', linewidth=1)

    # 绘制参考向量 (粗细和透明度由 weight 决定)
    # 根据用户要求，让箭头细一点、颜色浅一点
    linewidth = 0.5 + (w * 5)  # 降低基础粗细和缩放比例
    alpha_val = 0.2 + (w * 0.8)  # 降低基础透明度和缩放比例
    ax1.quiver(0, 0, dir_x, dir_y, angles='xy', scale_units='xy', scale=1.1,
               color=color, width=linewidth * 0.0015, alpha=min(1.0, alpha_val), zorder=4,
               label=f'{label} (自适应权重)')

    # 在向量末端标注 v_k
    ax1.text(dir_x * 0.95, dir_y * 0.95, label, fontsize=12, fontweight='bold', color=color,
             bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))

    # 散布模拟样本点 (距离原点有一定距离，且围绕各自的向量)
    if count > 0:
        spread = 0.12  # 角度散布范围
        r_points = np.random.uniform(0.3, 0.85, count)
        theta_points = np.random.normal(rad, spread, count)
        # 限制在第一象限
        theta_points = np.clip(theta_points, 0.05, np.pi / 2 - 0.05)
        px = r_points * np.cos(theta_points)
        py = r_points * np.sin(theta_points)

        # 统一使用灰色代表常规样本点
        sample_label = '种群点' if i == 0 else None
        ax1.scatter(px, py, color='#34495E', s=40, edgecolor='white', linewidth=0.8, alpha=0.7, zorder=5,
                    label=sample_label)

# 添加区域标注
ax1.text(0.65, 0.2, "密集区域\n（高密度，低权重）", fontsize=10,
         bbox=dict(facecolor='#EBF5FB', edgecolor='#3498DB', boxstyle='round,pad=0.4', alpha=0.9))
ax1.text(0.15, 0.75, "稀疏区域\n（低密度，高权重）", fontsize=10,
         bbox=dict(facecolor='#FDEDEC', edgecolor='#E74C3C', boxstyle='round,pad=0.4', alpha=0.9))

ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.set_aspect('equal')
ax1.set_xlabel('因子维度 1', fontweight='bold')
ax1.set_ylabel('因子维度 2', fontweight='bold')

# 图例设置在右上角
legend = ax1.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.95, borderpad=1)
legend.get_frame().set_edgecolor('#CCCCCC')

ax1.grid(True, linestyle='--', color='#E0E0E0', alpha=0.7, zorder=0)

plt.tight_layout()
plt.show()
