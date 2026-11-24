import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
file_path = "E1_20_best_history.xlsx - Sheet1.csv"
df = pd.read_csv(file_path)

# 2. 提取数据列
x = df.iloc[:, 0]  # 第一列：迭代次数
y = df.iloc[:, 1]  # 第二列：均匀性准则值

# 3. 设置中文字体（确保图表中的中文正常显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 4. 创建画布并绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='#1f77b4', linewidth=2, label='均匀性准则值')

# 5. 设置横坐标间隔为 300
max_x = int(x.max())
xticks = np.arange(0, max_x + 300, 300)
plt.xticks(xticks, fontsize=11)
plt.yticks(fontsize=11)

# 6. 设置图表标题和坐标轴标签
plt.xlabel('迭代次数 (Iteration)', fontsize=12)
plt.ylabel('均匀性准则值 (Criterion Value)', fontsize=12)
plt.title('优化收敛曲线', fontsize=14)

# 7. 开启网格线和图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 8. 保存图片
plt.savefig('convergence_curve.png', dpi=300, bbox_inches='tight')
plt.show()
