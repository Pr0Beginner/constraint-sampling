# IpoptConstraintHandler.py
import numpy as np
from scipy.optimize import minimize
from cyipopt import minimize_ipopt


class IpoptConstraintHandler:
    """
    约束处理工具类，提供等式和不等式约束下的采样方法，
    使用 IPOPT 作为求解器。
    """

    @staticmethod
    def find_diverse_feasible_point(initial_guess, encoder, constraints, existing_points, eps=1e-6):
        """
        使用 IPOPT 寻找一个可行点，该点与 'existing_points' 集合中所有点的最小距离最大化。

        Args:
            initial_guess (np.array): 优化的初始猜测点。
            encoder (FactorEncoder): 编码器实例。
            constraints (list): 约束列表。
            existing_points (list or np.array): 已知的可行点列表。
            eps (float): 约束检查的容忍度。

        Returns:
            np.array or None: 找到的可行点或None。
        """
        if not existing_points:
            return IpoptConstraintHandler.find_feasible_point(initial_guess, encoder, constraints, eps)

        existing_points_arr = np.array(existing_points)

        def objective(x):
            """
            目标函数：最小化 "与已知点集的最小距离的负数"。
            """
            distances_sq = np.sum((x - existing_points_arr) ** 2, axis=1)
            min_dist_sq = np.min(distances_sq)
            return -min_dist_sq

        def jacobian(x):
            """目标函数的雅可比矩阵（梯度）"""
            h = 1e-6
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += h
                grad[i] = (objective(x_plus) - objective(x)) / h
            return grad

        bounds = []
        for i in range(encoder.total_dims):
            info = encoder.factor_info_by_index[i]['info']
            if info['type'] == 'cont':
                bounds.append((0, 1))
            elif info['type'] == 'int':
                bounds.append((info['bounds'][0], info['bounds'][1]))
            else:
                bounds.append((0, len(info['options']) - 1))

        scipy_constraints = []
        for con in constraints:
            def constraint_fun(x, con=con):
                decoded = encoder.decode(x)
                return con['fun'](decoded)

            scipy_constraints.append({
                'type': con['type'],
                'fun': constraint_fun
            })

        result = minimize(
            objective,
            initial_guess,
            method=minimize_ipopt,
            jac=jacobian,
            constraints=scipy_constraints,
            bounds=bounds,
            options={'maxiter': 300, 'disp': False}
        )

        if result.success:
            final_point = encoder.clip_and_round(result.x)
            if IpoptConstraintHandler.is_feasible(final_point, encoder, constraints, eps):
                return final_point
        return None

    @staticmethod
    def find_feasible_point(initial_guess, encoder, constraints, eps=1e-6):
        """
        使用 IPOPT 从一个初始点出发，寻找一个满足所有约束的可行点。
        """

        def objective(x):
            """目标函数：最小化与初始猜测点的欧氏距离平方"""
            return np.sum((x - initial_guess) ** 2)

        def jacobian(x):
            """目标函数的雅可比矩阵（梯度）"""
            return 2 * (x - initial_guess)

        bounds = []
        for i in range(encoder.total_dims):
            info = encoder.factor_info_by_index[i]['info']
            if info['type'] == 'cont':
                bounds.append((0, 1))
            elif info['type'] == 'int':
                bounds.append((info['bounds'][0], info['bounds'][1]))
            else:
                bounds.append((0, len(info['options']) - 1))

        scipy_constraints = []
        for con in constraints:
            def constraint_fun(x, con=con):
                decoded = encoder.decode(x)
                return con['fun'](decoded)

            scipy_constraints.append({
                'type': con['type'],
                'fun': constraint_fun
            })

        result = minimize(
            objective,
            initial_guess,
            method=minimize_ipopt,
            jac=jacobian,
            constraints=scipy_constraints,
            bounds=bounds,
            options={'maxiter': 300, 'disp': False}
        )

        if result.success:
            final_point = encoder.clip_and_round(result.x)
            if IpoptConstraintHandler.is_feasible(final_point, encoder, constraints, eps):
                return final_point
        return None

    @staticmethod
    def is_feasible(sample, encoder, constraints, eps=1e-6):
        """检查样本是否满足所有约束"""
        decoded = encoder.decode(sample)
        for con in constraints:
            try:
                val = con['fun'](decoded)
                if con['type'] == 'ineq' and val < -eps:
                    return False
                if con['type'] == 'eq' and abs(val) > eps:
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def project_to_equality(sample, encoder, equality_constraints, eps=1e-6):
        """将样本投影到仅满足等式约束的曲面（用于变异后的快速修正）"""
        if not equality_constraints:
            return sample
        return IpoptConstraintHandler.find_feasible_point(sample, encoder, equality_constraints, eps)

    @staticmethod
    def check_pure_int_cat_constraints(encoder, constraints):
        """
        判断约束列表中哪些约束仅包含整数型(int)和分类变量(cat)
        """
        cont_vars = {
            var['name'] for var in encoder.factor_info_by_index.values()
            if var['info']['type'] == 'cont'
        }
        results = []
        for constraint in constraints:
            constraint_fun = constraint['fun']
            fun_str = str(constraint_fun)
            uses_cont_var = any(var in fun_str for var in cont_vars)
            results.append(not uses_cont_var)
        return results
