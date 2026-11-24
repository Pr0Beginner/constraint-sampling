# ConstraintHandler.py (Corrected Version)
import numpy as np
from scipy.optimize import minimize


class ConstraintHandler:
    """约束处理工具类，提供等式和不等式约束下的采样方法"""

    @staticmethod
    def find_diverse_feasible_point(initial_guess, encoder, constraints, existing_points, eps=1e-6):
        """
        使用 SLSQP 寻找一个可行点，该点与 'existing_points' 集合中所有点的最小距离最大化。

        Args:
            initial_guess (np.array): 优化的初始猜测点。
            encoder (FactorEncoder): 编码器实例。
            constraints (list): 约束列表。
            existing_points (list or np.array): 已知的可行点列表。
            eps (float): 约束检查的容忍度。

        Returns:
            np.array or None: 找到的可行点或None。
        """

        if not existing_points:  # 如果还没有任何可行点，则退化为寻找最近可行点
            return ConstraintHandler.find_feasible_point(initial_guess, encoder, constraints, eps)

        existing_points_arr = np.array(existing_points)

        def objective(x):
            """
            目标函数：最小化 "与已知点集的最小距离的负数"。
            即最大化与已知点集的最小距离。
            """
            # 计算 x 与所有已知点的欧氏距离的平方
            distances_sq = np.sum((x - existing_points_arr) ** 2, axis=1)
            # 找到最小距离的平方
            min_dist_sq = np.min(distances_sq)
            # 我们要最大化这个距离，所以最小化它的负值
            return -min_dist_sq

        # 变量边界约束 (与原函数相同)
        bounds = []
        for i in range(encoder.total_dims):
            info = encoder.factor_info_by_index[i]['info']
            if info['type'] == 'cont':
                bounds.append((0, 1))
            elif info['type'] == 'int':
                bounds.append((info['bounds'][0], info['bounds'][1]))
            else:  # cat
                bounds.append((0, len(info['options']) - 1))

        # 约束条件 (与原函数相同)
        scipy_constraints = []
        for con in constraints:
            def constraint_fun(x, con=con):
                decoded = encoder.decode(x)
                return con['fun'](decoded)

            scipy_constraints.append({
                'type': con['type'],
                'fun': constraint_fun
            })

        # 执行优化
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=scipy_constraints,
            bounds=bounds,
            options={'maxiter': 300, 'disp': False}
        )

        if result.success:
            final_point = encoder.clip_and_round(result.x)
            if ConstraintHandler.is_feasible(final_point, encoder, constraints, eps):
                return final_point
        return None

    @staticmethod
    def find_feasible_point(initial_guess, encoder, constraints, eps=1e-6):
        """
        使用 SLSQP 从一个初始点出发，寻找一个满足所有约束的可行点。
        这是处理可微分约束的核心方法。
        """

        # --- FIX 1: Correct objective function to MINIMIZE distance ---
        def objective(x):
            """目标函数：最小化与初始猜测点的欧氏距离平方"""
            return np.sum((x - initial_guess) ** 2)

        # 变量边界约束
        bounds = []
        for i in range(encoder.total_dims):
            info = encoder.factor_info_by_index[i]['info']
            if info['type'] == 'cont':
                bounds.append((0, 1))
            elif info['type'] == 'int':
                bounds.append((info['bounds'][0], info['bounds'][1]))
            else:  # cat
                bounds.append((0, len(info['options']) - 1))

        # --- FIX 2: Correctly handle BOTH equality and inequality constraints ---
        scipy_constraints = []
        for con in constraints:
            # 闭包问题：需要用 con=con 来捕获当前循环的 con
            def constraint_fun(x, con=con):
                decoded = encoder.decode(x)
                return con['fun'](decoded)

            scipy_constraints.append({
                'type': con['type'],  # 使用约束定义的类型
                'fun': constraint_fun
            })

        # 执行优化，SLSQP 擅长处理混合约束
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=scipy_constraints,
            bounds=bounds,
            options={'maxiter': 300, 'disp': False}
        )

        # 只有成功且最终点确实可行时，才返回结果
        if result.success:
            final_point = encoder.clip_and_round(result.x)
            if ConstraintHandler.is_feasible(final_point, encoder, constraints, eps):
                return final_point
        return None

    @staticmethod
    def is_feasible(sample, encoder, constraints, eps=1e-6):
        """检查样本是否满足所有约束"""
        decoded = encoder.decode(sample)
        for con in constraints:
            try:
                val = con['fun'](decoded)
                if con['type'] == 'ineq' and val < eps:
                    return False
                if con['type'] == 'eq' and abs(val) < eps:
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def project_to_equality(sample, encoder, equality_constraints, eps=1e-6):
        """将样本投影到仅满足等式约束的曲面（用于变异后的快速修正）"""
        if not equality_constraints:
            return sample
        # 调用通用函数，但只传入等式约束
        return ConstraintHandler.find_feasible_point(sample, encoder, equality_constraints, eps)

    @staticmethod
    def check_pure_int_cat_constraints(encoder, constraints):
        """
        判断约束列表中哪些约束仅包含整数型(int)和分类变量(cat)
        返回:
            list: 布尔值列表，与constraints对应，True表示是纯整数/分类变量约束
        """
        # 收集所有连续变量的名称
        cont_vars = {
            var['name'] for var in encoder.factor_info_by_index.values()
            if var['info']['type'] == 'cont'
        }
        # 用于存储每个约束的检查结果
        results = []
        for constraint in constraints:
            # 获取约束函数
            constraint_fun = constraint['fun']
            # 提取约束函数中使用的变量名
            fun_str = str(constraint_fun)
            # 检查是否使用了任何连续变量
            uses_cont_var = any(var in fun_str for var in cont_vars)
            # 如果没有使用连续变量，则认为是纯整数/分类变量约束
            results.append(not uses_cont_var)
        return results



