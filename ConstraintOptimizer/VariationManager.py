import numpy as np
from scipy.linalg import null_space

"""
交叉变异策略管理器
优化点：同步应用了底层的 CasADi 符号函数加速矩阵计算
"""
class VariationManager:
    def __init__(self, encoder, constraint_handler, uniformity_evaluator):
        self.encoder = encoder
        self.constraint_handler = constraint_handler
        self.uniformity_evaluator = uniformity_evaluator
        self.base_scale_factor = 0.5
        self.base_crossover_rate = 0.9

    def mutate(self, parents, population_type, constraint_tolerance=1e-6):
        if population_type == 'feasible':
            return self._feasible_mutation(parents, constraint_tolerance)
        return self._uniformity_mutation(parents, constraint_tolerance)

    def crossover(self, target, mutant, population_type):
        if population_type == 'feasible':
            return self._feasible_crossover(target, mutant)
        return self._uniformity_crossover(target, mutant)

    def _feasible_mutation(self, parents, constraint_tolerance):
        r1, r2, r3 = parents
        if abs(self.constraint_handler.unfeasible_compute(r1)) < constraint_tolerance:
            mutant = self._tangent_direction_mutation(r1, r2, r3)
        else:
            mutant = r1 + self.base_scale_factor * (r2 - r3)
        mutant = self.encoder.clip_and_round(mutant)
        if not self.constraint_handler.judge_feasible(mutant):
            repaired = self.constraint_handler.repair_infeasible_solution(mutant, [], constraint_tolerance)
            mutant = repaired if repaired is not None else r1
        return mutant

    def _uniformity_mutation(self, parents, constraint_tolerance):
        r1, r2, r3 = parents
        violation_r1 = self.constraint_handler.unfeasible_compute(r1)
        if abs(violation_r1) < constraint_tolerance:
            if np.random.rand() < 0.7:
                mutant = self._tangent_direction_mutation(r1, r2, r3)
            else: mutant = self._energy_guided_mutation(r1, parents)
        else:
            mutant = self._feasibility_guided_mutation(r1, r2, r3, violation_r1)
        return self.encoder.clip_and_round(mutant)

    def _tangent_direction_mutation(self, x, r2, r3):
        try:
            active_constraints = self._get_active_constraints(x)
            if not active_constraints:
                return x + self.base_scale_factor * (r2 - r3)
            J = self._compute_jacobian(x, active_constraints)
            Z = null_space(J)
            if Z.size == 0:
                return x + self.base_scale_factor * (r2 - r3)
            d = Z @ np.random.randn(Z.shape[1])
            d = d / (np.linalg.norm(d) + 1e-10)
            return x + self.base_scale_factor * np.random.uniform(0.5, 1.0) * d
        except Exception:
            return x + self.base_scale_factor * (r2 - r3)

    def _energy_guided_mutation(self, x, parents):
        energy_gradient = self._estimate_energy_gradient(x, parents)
        if energy_gradient is not None:
            return x - self.base_scale_factor * 0.1 * energy_gradient
        return x + self.base_scale_factor * (parents[1] - parents[2])

    def _feasibility_guided_mutation(self, x, r2, r3, violation):
        adaptive_scale = self.base_scale_factor * (1 + 0.5 * abs(violation))
        violation_r2 = self.constraint_handler.unfeasible_compute(r2)
        violation_r3 = self.constraint_handler.unfeasible_compute(r3)
        if abs(violation_r2) < self.constraint_handler.tolerance:
            return x + adaptive_scale * (r2 - x)
        elif abs(violation_r3) < self.constraint_handler.tolerance:
            return x + adaptive_scale * (r3 - x)
        return x + adaptive_scale * (r2 - r3)

    def _estimate_energy_gradient(self, x, parents, delta=1e-4):
        try:
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus, x_minus = x.copy(), x.copy()
                x_plus[i] += delta; x_minus[i] -= delta
                grad[i] = (self._calculate_point_energy(x_plus, parents) - self._calculate_point_energy(x_minus, parents)) / (2 * delta)
            return grad
        except Exception: return None

    def _calculate_point_energy(self, point, parents, p=2):
        total_energy = 0.0
        for parent in parents:
            dist = np.linalg.norm(point - parent)
            if dist > 0: total_energy += 1.0 / (dist ** p)
        return total_energy

    def _feasible_crossover(self, target, mutant):
        trial = target.copy()
        cross_points = np.random.rand(len(target)) < self.base_crossover_rate
        trial[cross_points] = mutant[cross_points]
        trial = self.encoder.clip_and_round(trial)
        if self.constraint_handler.unfeasible_compute(trial) <= self.constraint_handler.unfeasible_compute(target):
            return trial
        return target

    def _uniformity_crossover(self, target, mutant):
        target_violation = abs(self.constraint_handler.unfeasible_compute(target))
        adaptive_rate = self.base_crossover_rate * (1 + 0.2 * target_violation)
        trial = target.copy()
        cross_points = np.random.rand(len(target)) < adaptive_rate
        trial[cross_points] = mutant[cross_points]
        return self.encoder.clip_and_round(trial)

    def _get_active_constraints(self, x, epsilon=1e-6):
        """采用 CasADi JIT 加速活跃提取"""
        active = []
        if not self.constraint_handler.constraints: return active
        vals = self.constraint_handler.eval_con(x).full().flatten()
        for i, con in enumerate(self.constraint_handler.constraints):
            if con['type'] == 'eq' or (con['type'] == 'ineq' and abs(vals[i]) <= epsilon):
                active.append(i)
        return active

    def _compute_jacobian(self, x, active_constraints, delta=1e-6):
        """采用 CasADi JIT 的解析求导替代有限差分，提速近百倍"""
        if not active_constraints or not self.constraint_handler.constraints: return np.zeros((0, len(x)))
        J_all = self.constraint_handler.eval_jac(x).full()
        return J_all[active_constraints, :]
