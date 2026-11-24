import numpy as np
import casadi as ca

"""
约束处理工具类 (采用 CasADi 符号化加速版)
优化点：预构建了约束及其解析雅可比矩阵的符号函数，避免了运行时的循环差分计算。
"""


class ConstraintHandler:
    def __init__(self, encoder, constraints, tolerance=1e-8):
        """
        初始化约束处理器。
        """
        self.encoder = encoder
        self.constraints = constraints
        self.tolerance = tolerance

        # 定义 CasADi 的基础符号变量
        self.x_sym = ca.SX.sym('x', self.encoder.total_dims)

        # 1. 构建加速推断的符号函数
        self._build_fast_evaluators()

        # 2. 创建用于寻找最近可行点的主求解器 (基于 IPOPT)
        self._solver_dict = self._create_solver(tolerance=tolerance)

    def _build_fast_evaluators(self):
        """
        核心加速：将约束编译为 CasADi 符号图，包含函数求值和精确的解析雅可比矩阵
        """
        raw_g_list = []

        # 变量解码符号化
        decoded_sym = {}
        for name, idx in self.encoder.cont_pos.items():
            low, high = self.encoder.factors[name]['bounds']
            decoded_sym[name] = self.x_sym[idx] * (high - low) + low

        for name, info in self.encoder.disc_pos.items():
            decoded_sym[name] = self.x_sym[info['index']]

        # 遍历所有约束，构建原始表达式列表
        for con in self.constraints:
            val = con['fun'](decoded_sym)
            raw_g_list.append(val)

        if raw_g_list:
            # 拼接为向量并生成快速计算函数
            self.raw_g_sym = ca.vertcat(*raw_g_list)
            self.eval_con = ca.Function('eval_con', [self.x_sym], [self.raw_g_sym])

            # 自动微分求雅可比矩阵（解析梯度，极大提升速度和精度）
            self.raw_jac_sym = ca.jacobian(self.raw_g_sym, self.x_sym)
            self.eval_jac = ca.Function('eval_jac', [self.x_sym], [self.raw_jac_sym])
        else:
            self.raw_g_sym = ca.SX([])
            self.eval_con = None
            self.eval_jac = None

    def _create_solver(self, objective_builder=None, extra_constraints_builder=None, num_extra_vars=0, tolerance=1e-8):
        """
        创建 IPOPT 求解器
        """
        solver_options = {
            'ipopt': {'print_level': 0, 'sb': 'yes', 'max_iter': 1000, 'tol': tolerance},
            'print_time': 0
        }

        # 设置变量边界
        bounds_min, bounds_max = [], []
        for i in range(self.encoder.total_dims):
            info = self.encoder.factor_info_by_index[i]['info']
            if info['type'] == 'cont':
                bounds_min.append(0);
                bounds_max.append(1)
            elif info['type'] == 'int':
                bounds_min.append(info['bounds'][0]);
                bounds_max.append(info['bounds'][1])
            else:
                bounds_min.append(0);
                bounds_max.append(len(info['options']) - 1)

        c_sym = ca.SX.sym('c', self.encoder.total_dims)
        decision_vars = self.x_sym
        if num_extra_vars > 0:
            extra_vars_sym = ca.SX.sym('extra', num_extra_vars)
            decision_vars = ca.vertcat(self.x_sym, extra_vars_sym)

        # 构建 IPOPT 标准格式的约束
        g_ineq_list, g_eq_list = [], []
        if self.eval_con is not None:
            raw_expr = self.eval_con(self.x_sym)
            for i, con in enumerate(self.constraints):
                if con['type'] == 'ineq':
                    g_ineq_list.append(-raw_expr[i])  # IPOPT 要求 g <= 0
                elif con['type'] == 'eq':
                    g_eq_list.append(raw_expr[i])

        if extra_constraints_builder:
            extra_ineq, extra_eq = extra_constraints_builder(decision_vars)
            g_ineq_list.extend(extra_ineq)
            g_eq_list.extend(extra_eq)

        g_ineq = ca.vertcat(*g_ineq_list) if g_ineq_list else ca.SX([])
        g_eq = ca.vertcat(*g_eq_list) if g_eq_list else ca.SX([])
        g = ca.vertcat(g_ineq, g_eq)

        num_ineq_constraints = g_ineq.shape[0]
        num_eq_constraints = g_eq.shape[0]

        # 优化目标
        if objective_builder:
            objective = objective_builder(decision_vars, c_sym)
        else:
            objective = ca.sumsqr(self.x_sym - c_sym)

        nlp_prob = {'f': objective, 'x': decision_vars, 'g': g, 'p': c_sym}
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, solver_options)

        lbg = np.concatenate([-np.inf * np.ones(num_ineq_constraints), np.zeros(num_eq_constraints)])
        ubg = np.concatenate([np.zeros(num_ineq_constraints), np.zeros(num_eq_constraints)])

        return {
            'solver': solver,
            'lbx': np.array(bounds_min),
            'ubx': np.array(bounds_max),
            'lbg': lbg,
            'ubg': ubg
        }

    def find_feasible_point(self, initial_guess, eps=1e-6):
        try:
            sol = self._solver_dict['solver'](
                p=initial_guess,
                lbx=self._solver_dict['lbx'],
                ubx=self._solver_dict['ubx'],
                lbg=self._solver_dict['lbg'],
                ubg=self._solver_dict['ubg'],
                x0=initial_guess
            )
            final_point_rounded = self.encoder.clip_and_round(sol['x'].full().flatten())
            if self.judge_feasible(final_point_rounded):
                return final_point_rounded
            return None
        except Exception as e:
            return None

    def find_diverse_feasible_point(self, initial_guess, existing_points, eps=1e-6):
        if not existing_points:
            return self.find_feasible_point(initial_guess, eps)

        existing_points_arr = np.array(existing_points)

        def objective_builder(decision_vars, c_sym):
            return -decision_vars[-1]

        def extra_constraints_builder(decision_vars):
            x_sym = decision_vars[:-1]
            d_sym = decision_vars[-1]
            extra_ineq = [d_sym - ca.sumsqr(x_sym - p) for p in existing_points_arr]
            return extra_ineq, []

        diverse_solver_dict = self._create_solver(
            objective_builder=objective_builder,
            extra_constraints_builder=extra_constraints_builder,
            num_extra_vars=1
        )

        lbx_diverse = np.append(diverse_solver_dict['lbx'], 0)
        ubx_diverse = np.append(diverse_solver_dict['ubx'], np.inf)

        try:
            initial_guess_diverse = np.append(initial_guess, 0)
            sol = diverse_solver_dict['solver'](
                p=initial_guess,
                lbx=lbx_diverse,
                ubx=ubx_diverse,
                lbg=diverse_solver_dict['lbg'],
                ubg=diverse_solver_dict['ubg'],
                x0=initial_guess_diverse
            )
            final_point_rounded = self.encoder.clip_and_round(sol['x'].full().flatten()[:-1])
            if self.judge_feasible(final_point_rounded):
                return final_point_rounded
            return None
        except Exception:
            return self.find_feasible_point(initial_guess, eps)

    def repair_infeasible_solution(self, candidate_point, existing_points=None, eps=1e-6, max_attempts=5):
        if existing_points is None: existing_points = []

        if self.judge_feasible(candidate_point):
            return candidate_point

        repaired_point = self.find_feasible_point(candidate_point, eps)
        if repaired_point is not None: return repaired_point

        if existing_points:
            repaired_point = self.find_diverse_feasible_point(candidate_point, existing_points, eps)
            if repaired_point is not None: return repaired_point

        for attempt in range(max_attempts):
            perturbation = np.random.normal(0, 0.1, self.encoder.total_dims)
            perturbed_point = self.encoder.clip_and_round(candidate_point + perturbation)
            repaired_point = self.find_feasible_point(perturbed_point, eps)
            if repaired_point is not None: return repaired_point

            if existing_points:
                repaired_point = self.find_diverse_feasible_point(perturbed_point, existing_points, eps)
                if repaired_point is not None: return repaired_point
        return None

    def judge_feasible(self, sample):
        """极速判断可行性"""
        if not self.constraints: return True
        vals = self.eval_con(sample).full().flatten()
        for i, con in enumerate(self.constraints):
            if con['type'] == 'ineq' and vals[i] < -self.tolerance: return False
            if con['type'] == 'eq' and abs(vals[i]) > self.tolerance: return False
        return True

    def unfeasible_compute(self, sample):
        """极速计算约束违反度"""
        if not self.constraints: return 0.0
        vals = self.eval_con(sample).full().flatten()
        cv = 0.0
        for i, con in enumerate(self.constraints):
            if con['type'] == 'ineq' and vals[i] < 0:
                cv += -vals[i]
            elif con['type'] == 'eq':
                cv += abs(vals[i])
        return cv

    def cal_val(self, sin_sample):
        return self.unfeasible_compute(sin_sample)

    @staticmethod
    def is_feasible(sample, encoder, constraints, eps=1e-6):
        decoded = encoder.decode(sample)
        for con in constraints:
            try:
                val = con['fun'](decoded)
                if hasattr(val, 'full'): val = val.full().item()
                if con['type'] == 'ineq' and val < -eps: return False
                if con['type'] == 'eq' and abs(val) > eps: return False
            except Exception:
                return False
        return True

    def project_to_equality(self, sample, encoder, equality_constraints, eps=1e-6):
        if not equality_constraints: return sample
        temp_handler = ConstraintHandler(encoder, equality_constraints)
        return temp_handler.find_feasible_point(sample, eps)

    @staticmethod
    def check_pure_int_cat_constraints(encoder, constraints):
        cont_vars = {var['name'] for var in encoder.factor_info_by_index.values() if var['info']['type'] == 'cont'}
        results = []
        for constraint in constraints:
            fun_str = str(constraint['fun'])
            results.append(not any(var in fun_str for var in cont_vars))
        return results
