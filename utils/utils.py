import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyUtils:
    def __init__(self, encoder):
        self.encoder = encoder
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False

    def analyze_distribution(self, samples, title="因子分布分析"):
        """改进的可视化分析"""
        decoded = self.encoder.decode_to_arr(samples)
        factors = list(self.encoder.factors.items())

        n_factors = len(factors)
        cols = 2
        rows = int(np.ceil(n_factors / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        fig.suptitle(title, y=1.02, fontsize=14)

        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (name, info) in enumerate(factors):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            col_idx = list(self.encoder.factors.keys()).index(name)
            values = decoded[:, col_idx]

            if info['type'] == 'cont':
                values = values.astype(float)
                bins = np.linspace(min(values), max(values), len(samples) + 1)
                ax.hist(values, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, rwidth=1)
                ax.set_title(f"{name} 分布")
                ax.set_xlabel("值")
                ax.set_ylabel("频数")
                ax.set_xticks([])
                ax.grid(True, linestyle='--', alpha=0.6)
            elif info['type'] == 'int':
                values = values.astype(int)
                unique, counts = np.unique(values, return_counts=True)
                all_values = np.arange(info['bounds'][0], info['bounds'][1] + 1)
                full_counts = np.zeros_like(all_values)
                for uv, uc in zip(unique, counts):
                    full_counts[np.where(all_values == uv)[0][0]] = uc

                bars = ax.bar(all_values, full_counts, width=0.6, color='#2ca02c', edgecolor='white')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height, f"{int(height)}", ha='center', va='bottom',
                            fontsize=9)
                ax.set_xticks(all_values)
                ax.set_xlabel("Integer Values")
                ax.set_ylabel("Frequency")
                ax.grid(axis='y', alpha=0.3)
            else:
                unique, counts = np.unique(values, return_counts=True)
                y_pos = np.arange(len(unique))
                bars = ax.barh(y_pos, counts, height=0.6, color='#ff7f0e', edgecolor='white')
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height() / 2, f" {int(width)}", va='center', fontsize=9)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(unique)
                ax.invert_yaxis()
                ax.set_xlabel("Frequency")
                ax.grid(axis='x', alpha=0.3)

            ax.set_title(
                f"{name} ({'Continuous' if info['type'] == 'cont' else 'Integer' if info['type'] == 'int' else 'Categorical'})",
                pad=12, fontsize=12)

        for idx in range(n_factors, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_optimization_history(history, title='优化过程'):
        """绘制优化过程曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot([i for i in range(0, len(history))], history, marker='o', linestyle='--', color='r')
        plt.title(title)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值(越小越好)')
        plt.grid(True)
        plt.show()

    #
    # def export_to_excel(self, samples, file_path):
    #     """导出样本到Excel"""
    #     decoded_samples = [self.encoder.decode(sample) for sample in samples]
    #     df = pd.DataFrame(decoded_samples)
    #     df.to_excel(file_path, index=False)
    #     return file_path

    def print_results(self, samples):
        """控制台打印结果"""
        max_widths = {}
        header = []
        for name, info in self.encoder.factors.items():
            header_str = f"{name} ({info['type']})"
            header.append(header_str)
            max_widths[name] = len(header_str)

        for sample in samples:
            decoded = self.encoder.decode(sample)
            for name, info in self.encoder.factors.items():
                val = decoded[name]
                if info['type'] == 'cont':
                    formatted_val = f"{val:.4f}"
                elif info['type'] == 'int':
                    formatted_val = f"{int(val)}"
                else:
                    formatted_val = str(val)
                max_widths[name] = max(max_widths[name], len(formatted_val))

        print("\n======================= Factors =======================")
        formatted_header = " ".join(f"{h:<{max_widths[n]}}" for h, n in zip(header, self.encoder.factors.keys()))
        print(formatted_header)

        for sample in samples:
            decoded = self.encoder.decode(sample)
            row = []
            for name, info in self.encoder.factors.items():
                val = decoded[name]
                if info['type'] == 'cont':
                    row.append(f"{val:.4f}")
                elif info['type'] == 'int':
                    row.append(f"{int(val)}")
                else:
                    row.append(str(val))
            formatted_row = " ".join(f"{r:<{max_widths[n]}}" for r, n in zip(row, self.encoder.factors.keys()))
            print(formatted_row)

    @staticmethod
    def constraint_violation(optimizer, samples):
        """
        验证多个已解码样本是否满足约束条件 (修正版)
        - 等式约束 h(x)=0 允许在 [-eps, eps] 的容差范围内。
        - 不等式约束 g(x)>=0 必须严格满足。
        """
        violations = []
        details = []
        factor_names = list(optimizer.encoder.factors.keys())

        for idx, sample_arr in enumerate(samples):
            decoded = {name: sample_arr[i] for i, name in enumerate(factor_names)}
            total_violation_penalty = 0.0
            sample_details = {
                'sample_idx': idx,
                'is_feasible': True,
                'constraints': []
            }

            for con_idx, con in enumerate(optimizer.constraints):
                con_detail = {'constraint_idx': con_idx, 'type': con['type'], 'violation': 0.0, 'is_satisfied': True}
                try:
                    val = con['fun'](decoded)
                    if con['type'] == 'ineq':
                        # 不等式约束 g(x) >= 0，必须严格满足
                        if val < 0:
                            con_detail['violation'] = -val  # 违反值为正
                            con_detail['is_satisfied'] = False
                            total_violation_penalty += -val
                    elif con['type'] == 'eq':
                        # 等式约束 h(x) = 0，允许 eps 容差
                        abs_val = abs(val)
                        if abs_val > optimizer.eps:
                            con_detail['violation'] = abs_val - optimizer.eps
                            con_detail['is_satisfied'] = False
                            total_violation_penalty += abs_val

                    if not con_detail['is_satisfied']:
                        sample_details['is_feasible'] = False
                    sample_details['constraints'].append(con_detail)

                except Exception as e:
                    con_detail.update({'violation': float('inf'), 'is_satisfied': False, 'error': str(e)})
                    sample_details.update({'is_feasible': False, 'constraints': [con_detail]})
                    total_violation_penalty = float('inf')
                    break

            violations.append(total_violation_penalty)
            details.append(sample_details)

        return np.array(violations), details

    @staticmethod
    def isValidate(optimizer, result):
        """验证已解码样本并打印违反信息 (使用修正后的逻辑)"""
        if len(result) == 0:
            print("无样本可验证")
            return True

        violations, details = MyUtils.constraint_violation(optimizer, result)
        # 统计真正违反约束的样本数量
        violation_count = sum(1 for detail in details if not detail['is_feasible'])

        if violation_count > 0:
            print("\n--- 发现违反约束的样本 ---")
            for detail in details:
                if not detail['is_feasible']:
                    sample_values = result[detail['sample_idx']]
                    print(f"\n样本 {detail['sample_idx']}: {sample_values}")
                    for con in detail['constraints']:
                        if not con['is_satisfied']:
                            if 'error' in con:
                                print(f"  - 约束 {con['constraint_idx']} ({con['type']}): 执行错误 - {con['error']}")
                            else:
                                print(
                                    f"  - 约束 {con['constraint_idx']} ({con['type']}): 违反值 = {con['violation']:.6f}")
            print("\n--------------------------")

        print(f"\n违反约束总数: {violation_count}/{len(result)}")
        return violation_count == 0

    @staticmethod
    def export_result_to_excel(factors, result, excel_name):
        """
        将优化结果导出到Excel文件

        参数:
            optimizer: 优化器实例（包含factors和encoder）
            result: 优化器返回的结果数组（编码后的样本）
            excel_name: 导出的Excel文件名（包含.xlsx扩展名）
        """
        try:
            # 获取因子名称列表（保持定义时的顺序）
            factor_names = list(factors.keys())

            # 转换为DataFrame，使用因子名称作为列名
            df = pd.DataFrame(result, columns=factor_names)

            # 导出到Excel
            df.to_excel(excel_name, index=False)
            print(f"结果已成功导出到 {excel_name}")

        except Exception as e:
            print(f"导出Excel失败: {str(e)}")

    @staticmethod
    def draw_pro8(samples, color='green'):
        # 生成x1的取值范围
        x1 = np.linspace(0, 10, 500)
        # 计算约束g1的边界：x2 = x1^2 + 1
        x2_g1 = x1 ** 2 + 1
        # 生成x2的取值范围，用于计算约束g2的边界
        x2 = np.linspace(0, 10, 500)
        # 约束g2的边界：x1 = 1 + (x2 - 4)^2
        x1_g2 = 1 + (x2 - 4) ** 2
        # 绘制图形
        plt.figure(figsize=(8, 6))
        # 绘制约束g1的曲线
        plt.plot(x1, x2_g1, 'r-', label=r'$g_1(\vec{x}) = x_1^2 - x_2 + 1 = 0$')
        # 绘制约束g2的曲线
        plt.plot(x1_g2, x2, 'b-', label=r'$g_2(\vec{x}) = 1 - x_1 + (x_2 - 4)^2 = 0$')

        # 填充可行域（同时满足g1<=0和g2<=0的区域）
        # 思路：生成网格点，判断每个点是否满足两个约束，满足则标记为可行域
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)
        g1 = x1_mesh ** 2 - x2_mesh + 1
        g2 = 1 - x1_mesh + (x2_mesh - 4) ** 2
        feasible = (g1 <= 0) & (g2 <= 0)
        plt.contourf(x1_mesh, x2_mesh, feasible, levels=[0, 1], colors='green', alpha=0.3)

        for sample in samples:
            plt.scatter(
                sample[0],
                sample[1],
                color=color,
                marker='o',  # 形状：圆点
                s=50,  # 大小
            )

        # 图形设置
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.title('Feasible Region with Constraints')
        plt.show()

    @staticmethod
    def draw_pro8_comparison(last_population, population):
        """
        绘制 pro8 问题的两代种群对比图。
        - 上一代 (last_population) 为绿色圆圈
        - 当前代 (population) 为红色三角

        Args:
            last_population (np.ndarray): 解码后的上一代样本 (N, D)
            population (np.ndarray): 解码后的当前代样本 (M, D)
        """
        # --- 绘制约束边界 (来自 draw_pro8) ---
        x1 = np.linspace(0, 10, 500)
        x2_g1 = x1 ** 2 + 1  # g1
        x2 = np.linspace(0, 10, 500)
        x1_g2 = 1 + (x2 - 4) ** 2  # g2

        plt.figure(figsize=(8, 6))
        # 绘制约束曲线
        plt.plot(x1, x2_g1, 'r-', label=r'$g_1(\vec{x}) = x_1^2 - x_2 + 1 = 0$')
        plt.plot(x1_g2, x2, 'b-', label=r'$g_2(\vec{x}) = 1 - x_1 + (x_2 - 4)^2 = 0$')

        # 填充可行域
        x1_mesh, x2_mesh = np.meshgrid(x1, x2)
        g1 = x1_mesh ** 2 - x2_mesh + 1
        g2 = 1 - x1_mesh + (x2_mesh - 4) ** 2
        feasible = (g1 <= 0) & (g2 <= 0)
        plt.contourf(x1_mesh, x2_mesh, feasible, levels=[0, 1], colors='green', alpha=0.3)
        # --- 绘制约束边界结束 ---

        # 1. 绘制 last_population (绿色圆形)
        if last_population is not None and len(last_population) > 0:
            plt.scatter(
                last_population[:, 0],  # x1 坐标
                last_population[:, 1],  # x2 坐标
                color='green',
                marker='o',  # 圆形
                s=50,  # 大小
                label='上一代 (Last Pop)',
                alpha=0.8
            )

        # 2. 绘制 population (红色三角形)
        if population is not None and len(population) > 0:
            plt.scatter(
                population[:, 0],  # x1 坐标
                population[:, 1],  # x2 坐标
                color='red',
                marker='^',  # 三角形
                s=50,  # 大小
                label='当前代 (Current Pop)',
                alpha=0.8
            )

        # --- 图形设置 (来自 draw_pro8) ---
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.title('种群进化对比 (g08)')
        plt.show()

    @staticmethod
    def draw_g24_comparison(last_population, population, title='种群进化对比 (g24)'):
        """
        - 上一代 (last_population) 为绿色圆圈
        - 当前代 (population) 为红色三角

        Args:
            last_population (np.ndarray): 解码后的上一代样本 (N, 2)
            population (np.ndarray): 解码后的当前代样本 (M, 2)
            title (str): 图像标题
        """
        # g24 问题的特定边界
        x_min, x_max = 0, 3
        y_min, y_max = 0, 4

        # --- 绘制约束边界 ---
        # 1. 创建 x1 向量
        x1 = np.linspace(x_min, x_max, 500)

        # 2. 计算 g1 边界: x2 = 2*x1**4 - 8*x1**3 + 8*x1**2 + 2
        x2_g1 = 2 * x1 ** 4 - 8 * x1 ** 3 + 8 * x1 ** 2 + 2

        # 3. 计算 g2 边界: x2 = 4*x1**4 - 32*x1**3 + 88*x1**2 - 96*x1 + 36
        x2_g2 = 4 * x1 ** 4 - 32 * x1 ** 3 + 88 * x1 ** 2 - 96 * x1 + 36

        plt.figure(figsize=(8, 6))

        # 4. 绘制约束曲线
        # (我们只绘制在 [ymin, ymax] 范围内的部分)
        plt.plot(x1, x2_g1, 'r-', label=r'$g_1(\vec{x})=0$ (boundary)')
        plt.plot(x1, x2_g2, 'b-', label=r'$g_2(\vec{x})=0$ (boundary)')

        # --- 填充可行域 ---
        # 1. 创建网格
        x1_mesh, x2_mesh = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        # 2. 计算 g1 和 g2 在网格上的值 (使用原始 <= 0 的形式)
        # g1: -2*x1**4 + 8*x1**3 - 8*x1**2 + x2 - 2 <= 0
        g1_vals = -2 * x1_mesh ** 4 + 8 * x1_mesh ** 3 - 8 * x1_mesh ** 2 + x2_mesh - 2

        # g2: -4*x1**4 + 32*x1**3 - 88*x1**2 + 96*x1 + x2 - 36 <= 0
        g2_vals = -4 * x1_mesh ** 4 + 32 * x1_mesh ** 3 - 88 * x1_mesh ** 2 + 96 * x1_mesh + x2_mesh - 36

        # 3. 找到可行区域 (g1 <= 0 且 g2 <= 0)
        feasible = (g1_vals <= 0) & (g2_vals <= 0)

        # 4. 填充
        plt.contourf(x1_mesh, x2_mesh, feasible, levels=[0, 1], colors='green', alpha=0.3)

        # --- 绘制样本点 (逻辑与 draw_pro8_comparison 相同) ---
        # 1. 绘制 last_population (绿色圆形)
        if last_population is not None and len(last_population) > 0:
            plt.scatter(
                last_population[:, 0],  # x1 坐标
                last_population[:, 1],  # x2 坐标
                color='green',
                marker='o',  # 圆形
                s=50,  # 大小
                label='上一代 (Last Pop)',
                alpha=0.8
            )

        # 2. 绘制 population (红色三角形)
        if population is not None and len(population) > 0:
            plt.scatter(
                population[:, 0],  # x1 坐标
                population[:, 1],  # x2 坐标
                color='red',
                marker='^',  # 三角形
                s=50,  # 大小
                label='当前代 (Current Pop)',
                alpha=0.8
            )

        # --- 图形设置 ---
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.title(title)
        plt.show()

    @staticmethod
    def draw_g24_samples(samples, title='g24 约束条件与样本点'):
        """
        绘制 g24 问题的可行域及一组样本点。
        - 样本点 (samples) 会被绘制为红色三角

        Args:
            samples (np.ndarray): 解码后的样本 (N, 2)
            title (str): 图像标题
        """
        # g24 问题的特定边界
        x_min, x_max = 0, 3
        y_min, y_max = 0, 4

        # --- 绘制约束边界 ---
        # 1. 创建 x1 向量
        x1 = np.linspace(x_min, x_max, 500)

        # 2. 计算 g1 边界: x2 = 2*x1**4 - 8*x1**3 + 8*x1**2 + 2
        x2_g1 = 2 * x1 ** 4 - 8 * x1 ** 3 + 8 * x1 ** 2 + 2

        # 3. 计算 g2 边界: x2 = 4*x1**4 - 32*x1**3 + 88*x1**2 - 96*x1 + 36
        x2_g2 = 4 * x1 ** 4 - 32 * x1 ** 3 + 88 * x1 ** 2 - 96 * x1 + 36

        plt.figure(figsize=(8, 6))

        # 4. 绘制约束曲线
        plt.plot(x1, x2_g1, 'r-', label=r'$g_1(\vec{x})=0$ (boundary)')
        plt.plot(x1, x2_g2, 'b-', label=r'$g_2(\vec{x})=0$ (boundary)')

        # --- 填充可行域 ---
        # 1. 创建网格
        x1_mesh, x2_mesh = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

        # 2. 计算 g1 和 g2 在网格上的值 (使用原始 <= 0 的形式)
        # g1: -2*x1**4 + 8*x1**3 - 8*x1**2 + x2 - 2 <= 0
        g1_vals = -2 * x1_mesh ** 4 + 8 * x1_mesh ** 3 - 8 * x1_mesh ** 2 + x2_mesh - 2

        # g2: -4*x1**4 + 32*x1**3 - 88*x1**2 + 96*x1 + x2 - 36 <= 0
        g2_vals = -4 * x1_mesh ** 4 + 32 * x1_mesh ** 3 - 88 * x1_mesh ** 2 + 96 * x1_mesh + x2_mesh - 36

        # 3. 找到可行区域 (g1 <= 0 且 g2 <= 0)
        feasible = (g1_vals <= 0) & (g2_vals <= 0)

        # 4. 填充
        plt.contourf(x1_mesh, x2_mesh, feasible, levels=[0, 1], colors='green', alpha=0.3)

        # --- 绘制样本点 (您代码中的逻辑) ---
        if samples is not None and len(samples) > 0:
            plt.scatter(
                samples[:, 0],  # x1 坐标
                samples[:, 1],  # x2 坐标
                color='red',
                marker='^',  # 三角形
                s=50,  # 大小
                label='样本点 (Samples)',  # 标签已修正
                alpha=0.8
            )

        # --- 图形设置 ---
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.title(title)  # 使用函数参数作为标题
        plt.show()

    @staticmethod
    def draw_pro8_movement_arrows(decoded_pairs, title="g08 变异与交叉"):
        """
        使用箭头 (Quiver) 可视化 (父代 -> 子代) 的移动。
        - 父代 (起始点) 为绿色圆圈
        - 子代 (终止点) 为红色三角
        - 移动路径为灰色箭头

        Args:
            decoded_pairs (list): 包含 (父代_array, 子代_array) 元组的列表
            title (str): 图像标题
        """
        # --- 绘制约束边界 (同上) ---
        x1 = np.linspace(0, 10, 500)
        x2_g1 = x1 ** 2 + 1  # g1
        x2 = np.linspace(0, 10, 500)
        x1_g2 = 1 + (x2 - 4) ** 2  # g2

        plt.figure(figsize=(10, 8))  # 稍微调大图像以便看清
        plt.plot(x1, x2_g1, 'r-', label=r'$g_1(\vec{x}) = x_1^2 - x_2 + 1 = 0$')
        plt.plot(x1_g2, x2, 'b-', label=r'$g_2(\vec{x}) = 1 - x_1 + (x_2 - 4)^2 = 0$')

        x1_mesh, x2_mesh = np.meshgrid(x1, x2)
        g1 = x1_mesh ** 2 - x2_mesh + 1
        g2 = 1 - x1_mesh + (x2_mesh - 4) ** 2
        feasible = (g1 <= 0) & (g2 <= 0)
        plt.contourf(x1_mesh, x2_mesh, feasible, levels=[0, 1], colors='green', alpha=0.3)
        # --- 约束边界结束 ---

        if not decoded_pairs:
            print("警告: 没有可绘制的变异配对。")
            return

        # 1. 提取所有父代和子代点
        parents = np.array([p for p, c in decoded_pairs])
        children = np.array([c for p, c in decoded_pairs])

        # 2. 绘制父代点 (绿色圆圈)
        plt.scatter(
            parents[:, 0], parents[:, 1],
            color='green', marker='o', s=50,
            label='父代 (Parent)', alpha=0.8, zorder=2
        )

        # 3. 绘制子代点 (红色三角)
        plt.scatter(
            children[:, 0], children[:, 1],
            color='red', marker='^', s=50,
            label='子代 (Child)', alpha=0.8, zorder=3
        )

        # 4. 绘制箭头 (Quiver)
        # 计算每个点的 x 和 y 方向的“移动”
        dx = children[:, 0] - parents[:, 0]
        dy = children[:, 1] - parents[:, 1]

        # 绘制箭头
        plt.quiver(
            parents[:, 0], parents[:, 1],  # 箭头起始点
            dx, dy,  # 箭头向量
            angles='xy', scale_units='xy', scale=1,  # 确保箭头按数据坐标1:1绘制
            color='gray',
            alpha=0.7,
            width=0.003,  # 箭头宽度
            zorder=1  # 放在点后面
        )

        # --- 图形设置 ---
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.title(title)
        plt.show()

    @staticmethod
    def add_unique_array(main_list, new_array, seen_set):
        """
        向二维数组中添加一维数组，确保不重复
        :param main_list: 存储一维数组的二维数组（主列表）
        :param new_array: 待添加的一维数组
        :param seen_set: 用于记录已添加数组的集合（存储元组）
        :return: 是否成功添加（True/False）
        """
        # 将一维数组转为元组（可哈希，用于集合判断）
        new_tuple = tuple(new_array)
        if new_tuple not in seen_set:
            main_list.append(new_array)  # 添加原数组到二维数组
            seen_set.add(new_tuple)  # 记录已添加的元组
            return True
        return False
