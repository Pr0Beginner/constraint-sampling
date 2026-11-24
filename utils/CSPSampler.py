import constraint
import numpy as np

class CSPSampler:
    """
    使用CSP (Constraint Satisfaction Problem) 求解器来寻找满足约束的、唯一的离散样本点。
    """
    @staticmethod
    def find_feasible_solutions(factors, constraints, num_samples):
        """
        寻找满足所有约束的唯一可行解。

        返回:
            - solutions: 一个包含可行解字典的列表，例如 [{'x': 1, 'y': 'A'}, ...]
            - success: 一个布尔值，指示是否成功找到了足够数量的解
        """
        problem = constraint.Problem()

        # 1. 添加变量及其定义域 (domains)
        discrete_factor_names = []
        for name, info in factors.items():
            if info['type'] == 'int':
                problem.addVariable(name, range(info['bounds'][0], info['bounds'][1] + 1))
                discrete_factor_names.append(name)
            elif info['type'] == 'cat':
                problem.addVariable(name, info['options'])
                discrete_factor_names.append(name)
            else:
                # CSP 求解器主要用于离散变量，如果存在连续变量，则此方法不适用
                print(f"警告：CSPSampler 仅适用于纯离散问题。")
                return [], False

        # 2. 添加约束
        for con in constraints:
            # 该库要求约束函数只接受变量作为参数，而不是一个字典
            def create_constraint_wrapper(constraint_func, factor_order):
                def wrapper(*args):
                    # 将传入的参数重新组合成优化器期望的字典格式
                    v = {name: val for name, val in zip(factor_order, args)}

                    # 根据约束类型返回 True/False
                    result = constraint_func(v)
                    if con['type'] == 'eq':
                        return result == 0
                    elif con['type'] == 'ineq':
                        return result >= 0
                    return False

                return wrapper

            # 使用包装函数添加约束
            problem.addConstraint(
                create_constraint_wrapper(con['fun'], discrete_factor_names),
                discrete_factor_names
            )

        # 3. 求解并获取所有唯一解
        print("CSP求解器开始寻找所有唯一可行解...")
        all_solutions = problem.getSolutions()
        print(f"求解完成，共找到 {len(all_solutions)} 个唯一可行解。")

        if not all_solutions:
            return [], False

        # 4. 为避免样本顺序偏好，随机打乱所有解
        np.random.shuffle(all_solutions)

        # 5. 根据需求返回指定数量的样本
        if len(all_solutions) < num_samples:
            print(
                f"警告：找到的唯一解数量 ({len(all_solutions)}) 少于请求的样本数量 ({num_samples})。将返回所有找到的解。")
            return all_solutions, True  # 即使数量不足，也认为是成功了

        return all_solutions[:num_samples], True
