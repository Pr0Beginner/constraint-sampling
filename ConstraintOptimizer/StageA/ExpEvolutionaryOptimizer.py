import numpy as np
from scipy.stats import qmc
from scipy.linalg import null_space
from scipy.spatial.distance import pdist, squareform

if __package__:
    from .VariationManager import VariationManager
    from .FactorEncoder import FactorEncoder
    from .ConstraintHandler import ConstraintHandler
else:
    from VariationManager import VariationManager
    from FactorEncoder import FactorEncoder
    from ConstraintHandler import ConstraintHandler

from Criterion.PhiPCriterion import PhiPCriterion


class EvolutionaryOptimizer:
    def __init__(self, factors, constraints, sample_num, max_generations=100, tolerance=1e-6, isDraw=False):
        self.sample_num = sample_num
        self.max_generations = max_generations
        self.tolerance = tolerance
        self.isDraw = isDraw

        self.encoder = FactorEncoder(factors)
        self.constraint_handler = ConstraintHandler(self.encoder, constraints, tolerance)
        self.uniformity_evaluator = PhiPCriterion(p_exponent=2, p_norm=2)

        self.start_pop_mult = 4.0
        self.turn_pop_mult = 1.2
        self.turn_point_ratio = 0.9
        self.current_pop_size = int(self.sample_num * self.start_pop_mult)

        self.history = []
        n_vecs = self.sample_num
        self.reference_vectors = self._generate_reference_vectors(n_vecs)
        self.ref_vector_weights = np.ones(n_vecs) / n_vecs
        self.ref_vector_success = np.zeros(n_vecs)
        self.ref_vector_attempts = np.zeros(n_vecs)
        self.ema_learning_rate = 0.1
        self.global_step_factor = 1.0
        self.stagnation_counter = 0
        self.F_value = 0.5
        self.feasible_set = set()
        self.feasible_arr = []
        self.elite_population = []

        self.variation_manager = VariationManager(
            self.encoder, self.constraint_handler, self.uniformity_evaluator
        )

    def _evaluate_population_stats(self, population):
        feasible_mask = np.zeros(len(population), dtype=bool)
        cv_values = np.zeros(len(population), dtype=float)
        for i, p in enumerate(population):
            feasible, cv = self.constraint_handler.evaluate_feasibility_and_cv(p)
            feasible_mask[i] = feasible
            cv_values[i] = cv
        return feasible_mask, cv_values

    def optimize(self, use_repulsion = True, use_control = True):
        population = self._initialize_population_dynamic()
        decoded_elite = self.encoder.decode_to_number(np.array(self.elite_population))
        self.best_val = self.uniformity_evaluator.evaluate(decoded_elite)
        self.history.append(self.best_val)

        for gen in range(self.max_generations):
            target_size = self._get_dynamic_pop_size(gen)
            self.current_pop_size = target_size

            pop_feasible_mask, _ = self._evaluate_population_stats(population)
            feasible_indices = np.flatnonzero(pop_feasible_mask)
            self._update_reference_weights(population, pop_feasible_mask)

            offspring = []
            feasible_offspring = []
            for i, ind in enumerate(population):
                target_vec, ref_idx = self._select_target_vector(ind, population, feasible_indices)
                # 基于局部能量梯度排斥的进化方向引导方法
                if use_repulsion:
                    target_vec = self._apply_repulsion_adjustment(ind, target_vec, population, i, gen)

                if np.random.rand() < 0.3:
                    ind = self._get_repulsive_shifted_point(ind, population, i, gen)
                child = ind
                if use_control:
                    # 基于切空间映射与非线性求解器的约束边界控制策略
                    if self.constraint_handler.judge_feasible(ind):
                        child = self._evolve_feasible_on_manifold(ind, target_vec)
                    else:
                        child = self._evolve_infeasible(ind, target_vec)

                is_child_feasible = self.constraint_handler.judge_feasible(child)
                if ref_idx is not None and is_child_feasible:
                    self.ref_vector_success[ref_idx] += 1
                offspring.append(child)

                if is_child_feasible:
                    feasible_offspring.append(child)

            if feasible_offspring:
                self._improve_elites_by_swap(feasible_offspring)

            self.history.append(self.best_val)
            population_pool = np.vstack([population, offspring])
            population = self._survival_selection(population_pool, target_size)
            self._update_global_step_factor()

        return self.encoder.decode_to_number(np.array(self.elite_population))

    def _get_repulsive_shifted_point(self, current_point, population, current_idx, gen):
        k_neighbors = 3
        base_shift_strength = 0.05
        min_dist_eps = 1e-7
        progress = gen / self.max_generations
        decay_factor = max(0.0, 1.0 - progress * 1.2)
        if decay_factor < 1e-3:
            return current_point.copy()

        diffs = population - current_point
        dists_sq = np.sum(diffs ** 2, axis=1)
        dists_sq[current_idx] = np.inf
        neighbor_indices = np.argsort(dists_sq)[:k_neighbors]

        repulsion_force = np.zeros_like(current_point)
        p_exp = 2.0
        found_close_neighbor = False
        for idx in neighbor_indices:
            vec_away = -diffs[idx]
            dist = np.sqrt(dists_sq[idx])
            if dist < min_dist_eps:
                vec_away = np.random.randn(len(current_point))
                dist = min_dist_eps
            weight = 1.0 / (dist ** p_exp)
            repulsion_force += (vec_away / dist) * weight
            found_close_neighbor = True

        if not found_close_neighbor:
            return current_point.copy()

        force_norm = np.linalg.norm(repulsion_force)
        if force_norm > 1e-9:
            force_dir = repulsion_force / force_norm
        else:
            return current_point.copy()

        step_size = base_shift_strength * decay_factor * self.global_step_factor
        return self.encoder.clip_and_round(current_point + force_dir * step_size)

    # 基于局部能量梯度排斥的进化方向引导方法
    def _apply_repulsion_adjustment(self, current_point, target_vec, population, current_idx, gen):
        k_neighbors = 5
        min_dist_eps = 1e-7
        progress = gen / self.max_generations
        alpha = 0.5 * (1.0 - progress) + 0.05

        original_dir = target_vec - current_point
        orig_norm = np.linalg.norm(original_dir)
        if orig_norm < 1e-9:
            return target_vec
        original_dir_normed = original_dir / orig_norm

        diffs = population - current_point
        dists = np.linalg.norm(diffs, axis=1)
        dists[current_idx] = np.inf
        neighbor_indices = np.argsort(dists)[:k_neighbors]

        repulsion_vec = np.zeros_like(current_point)
        p_exp = self.uniformity_evaluator.p_exponent

        for idx in neighbor_indices:
            dist = max(dists[idx], min_dist_eps)
            dir_away = -diffs[idx]
            weight = 1.0 / (dist ** (p_exp + 1))
            repulsion_vec += dir_away * weight

        rep_norm = np.linalg.norm(repulsion_vec)
        repulsion_vec_normed = repulsion_vec / rep_norm if rep_norm > 1e-9 else np.zeros_like(current_point)

        combined_dir = (1 - alpha) * original_dir_normed + alpha * repulsion_vec_normed
        comb_norm = np.linalg.norm(combined_dir)
        if comb_norm > 1e-9:
            combined_dir /= comb_norm

        return current_point + combined_dir * orig_norm

    def _improve_elites_by_swap(self, candidates):
        current_elites = np.array(self.elite_population.copy())
        decoded = self.encoder.decode_to_number(current_elites)
        cur_val = self.uniformity_evaluator.evaluate(decoded)
        is_updated = False

        for sample in candidates:
            temp = current_elites.copy()
            sel_index = np.random.randint(0, self.sample_num, 10)
            for index in sel_index:
                temp[index] = sample
                temp_val = self.uniformity_evaluator.evaluate(self.encoder.decode_to_number(np.array(temp)))
                if temp_val < cur_val:
                    current_elites = temp
                    is_updated = True
                    cur_val = temp_val
                    break

        if is_updated:
            self.elite_population = current_elites
            self.best_val = self.uniformity_evaluator.evaluate(
                self.encoder.decode_to_number(np.array(self.elite_population))
            )

    def _get_dynamic_pop_size(self, gen):
        T = self.max_generations
        N_start = int(self.sample_num * self.start_pop_mult)
        N_turn = int(self.sample_num * self.turn_pop_mult)
        N_final = self.sample_num
        T_turn = int(T * self.turn_point_ratio)
        if gen < T_turn:
            return max(N_final, int(N_start - ((N_start - N_turn) / T_turn) * gen))
        if T == T_turn:
            return N_final
        return max(N_final, int(N_turn - ((N_turn - N_final) / (T - T_turn)) * (gen - T_turn)))

    def _survival_selection(self, pool, target_size):
        next_gen = [row.copy() for row in self.elite_population]
        needed = target_size - len(next_gen)
        if needed <= 0:
            return np.array(next_gen[:target_size])

        infeasible_candidates = []
        infeasible_cvs = []
        for p in pool:
            feasible, cv = self.constraint_handler.evaluate_feasibility_and_cv(p)
            if not feasible:
                infeasible_candidates.append(p)
                infeasible_cvs.append(abs(cv))

        if infeasible_candidates:
            sorted_indices = np.argsort(infeasible_cvs)
            take_num = min(needed, len(infeasible_candidates))
            next_gen.extend([infeasible_candidates[i] for i in sorted_indices[:take_num]])
            needed -= take_num

        while needed > 0:
            next_gen.append(np.random.rand(self.encoder.total_dims))
            needed -= 1
        return np.array(next_gen)

    def _filter_by_phip_greedy(self, points, k):
        current_points = points.copy()
        if len(current_points) <= k:
            return current_points
        p_exp = self.uniformity_evaluator.p_exponent
        while len(current_points) > k:
            dists = squareform(pdist(current_points))
            np.fill_diagonal(dists, np.inf)
            energies = np.sum(dists ** (-p_exp), axis=1)
            worst_idx = np.argmax(energies)
            current_points = np.delete(current_points, worst_idx, axis=0)
        return current_points

    def _initialize_population_dynamic(self):
        sampler = qmc.LatinHypercube(d=self.encoder.total_dims)
        raw_samples = sampler.random(int(self.current_pop_size * 1.5))
        population = self.encoder.clip_and_round(raw_samples)

        initialized_pop, all_feasible_found = [], []

        for p in population:
            if self.constraint_handler.judge_feasible(p):
                initialized_pop.append(p)
                all_feasible_found.append(p)
            else:
                repaired = self.constraint_handler.repair_infeasible_solution(p, all_feasible_found, self.tolerance)
                if repaired is not None:
                    initialized_pop.append(repaired)
                    all_feasible_found.append(repaired)
                else:
                    initialized_pop.append(p)
            if len(initialized_pop) >= int(self.current_pop_size * 1.2):
                break

        max_attempts = self.sample_num * 20
        attempts = 0

        while len(all_feasible_found) < self.sample_num and attempts < max_attempts:
            attempts += 1
            repaired = self.constraint_handler.find_feasible_point(
                np.random.rand(self.encoder.total_dims), eps=self.tolerance
            )
            if repaired is not None:
                all_feasible_found.append(repaired)
                initialized_pop.append(repaired)

        if len(all_feasible_found) < self.sample_num:
            needed = self.sample_num - len(all_feasible_found)
            print(
                f"  [提示] 初始可行域极其狭窄，当前仅找到 {len(all_feasible_found)} 个可行解。"
                f"剩余 {needed} 个空位将用随机解填充，由后续进化修复。"
            )
            for _ in range(needed):
                rand_p = np.random.rand(self.encoder.total_dims)
                all_feasible_found.append(rand_p)
                initialized_pop.append(rand_p)

        self.elite_population = self._initial_filter(np.array(all_feasible_found), self.sample_num)

        if len(initialized_pop) > self.current_pop_size:
            initialized_pop = initialized_pop[:self.current_pop_size]
        while len(initialized_pop) < self.current_pop_size:
            initialized_pop.append(np.random.rand(self.encoder.total_dims))

        return np.array(initialized_pop)

    def _initial_filter(self, points, k):
        current_points = points.copy()
        if len(current_points) <= k:
            return list(current_points)
        p_exp = self.uniformity_evaluator.p_exponent
        while len(current_points) > k:
            dists = squareform(pdist(current_points))
            np.fill_diagonal(dists, np.inf)
            energies = np.sum(dists ** (-p_exp), axis=1)
            current_points = np.delete(current_points, np.argmax(energies), axis=0)
        return list(current_points)

    def _generate_reference_vectors(self, n_vecs):
        vecs = np.random.normal(0, 1, (n_vecs, self.encoder.total_dims))
        return vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]

    def _update_reference_weights(self, population, feasible_mask=None):
        if len(population) == 0:
            return

        if feasible_mask is None:
            feasible_mask, _ = self._evaluate_population_stats(population)
        if not np.any(feasible_mask):
            return

        feasible_pop = np.asarray(population)[feasible_mask]
        counts = np.zeros(len(self.reference_vectors))
        pop_centered = feasible_pop - np.mean(feasible_pop, axis=0)

        for ind_centered in pop_centered:
            norm = np.linalg.norm(ind_centered)
            if norm >= 1e-9:
                best_vec_idx = np.argmax(np.dot(self.reference_vectors, ind_centered) / norm)
                counts[best_vec_idx] += 1

        W_sparse = 1.0 / (counts + 1.0)
        W_history = 0.5 + (self.ref_vector_success / (self.ref_vector_attempts + 1e-6))
        current_score = W_sparse * W_history
        new_weights = (1 - self.ema_learning_rate) * self.ref_vector_weights + self.ema_learning_rate * current_score
        self.ref_vector_weights = new_weights / np.sum(new_weights)

    def _select_target_vector(self, ind, population, feasible_indices=None):
        if feasible_indices is None:
            feasible_indices = np.array(
                [i for i, p in enumerate(population) if self.constraint_handler.judge_feasible(p)],
                dtype=int,
            )
        if np.random.rand() < 0.2 and feasible_indices.size > 0:
            return population[np.random.choice(feasible_indices)], None

        idx = np.random.choice(len(self.reference_vectors), p=self.ref_vector_weights)
        self.ref_vector_attempts[idx] += 1
        return self.reference_vectors[idx] + ind, idx

    def _update_global_step_factor(self):
        if len(self.history) > 10:
            if abs(self.history[-10] - self.history[-1]) < 1e-7:
                self.stagnation_counter += 1
                self.global_step_factor = max(0.01, self.global_step_factor * 0.9)
            else:
                self.stagnation_counter = 0
                self.global_step_factor = min(1.0, self.global_step_factor * 1.05)

    def _get_active_constraints(self, x, epsilon=1e-5):
        active = []
        if not self.constraint_handler.constraints:
            return active
        vals = self.constraint_handler.eval_con(x).full().flatten()
        for i, con in enumerate(self.constraint_handler.constraints):
            if con["type"] == "eq" or (con["type"] == "ineq" and abs(vals[i]) < epsilon):
                active.append(i)
        return active

    def _get_violated_constraints_idx(self, x):
        violated = []
        if not self.constraint_handler.constraints:
            return violated
        vals = self.constraint_handler.eval_con(x).full().flatten()
        for i, con in enumerate(self.constraint_handler.constraints):
            if con["type"] == "ineq" and vals[i] < -self.tolerance:
                violated.append(i)
            elif con["type"] == "eq" and abs(vals[i]) > self.tolerance:
                violated.append(i)
        return violated

    def _compute_jacobian(self, x, active_idx, delta=1e-6):
        if not active_idx or not self.constraint_handler.constraints:
            return np.zeros((0, len(x)))
        J_all = self.constraint_handler.eval_jac(x).full()
        return J_all[active_idx, :]

    def _get_feasibility_margin(self, current_point):
        if not self.constraint_handler.constraints:
            return 1.0
        vals = self.constraint_handler.eval_con(current_point).full().flatten()
        ineq_vals = [vals[i] for i, con in enumerate(self.constraint_handler.constraints) if con["type"] == "ineq"]
        return max(0.0, min(ineq_vals)) if ineq_vals else 1.0

    def _evolve_feasible_on_manifold(self, current_point, target_vec):
        min_step = 1e-3
        active_constraints = self._get_active_constraints(current_point)

        if active_constraints:
            J = self._compute_jacobian(current_point, active_constraints)
            try:
                Z = null_space(J)
                if Z.size > 0:
                    desired_dir = target_vec - current_point
                    tangent_dir = Z @ (Z.T @ desired_dir)
                    if np.linalg.norm(tangent_dir) < 1e-6:
                        tangent_dir = Z @ np.random.randn(Z.shape[1])
                    direction = tangent_dir
                else:
                    direction = np.zeros_like(current_point)
            except Exception:
                direction = np.zeros_like(current_point)
            step_size = (0.05 * self.global_step_factor) + min_step
        else:
            direction = target_vec - current_point
            margin = self._get_feasibility_margin(current_point)
            step_size = (0.2 * self.global_step_factor * np.tanh(margin * 5.0)) + min_step

        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction = direction / norm

        return self.encoder.clip_and_round(current_point + step_size * direction)

    def _evolve_infeasible(self, current_point, target_vec):
        F_search = target_vec - current_point
        norm_search = np.linalg.norm(F_search)
        if norm_search > 1e-9:
            F_search /= norm_search

        F_repair = np.zeros_like(current_point)
        violated_idx = self._get_violated_constraints_idx(current_point)
        if violated_idx:
            J_violated = self._compute_jacobian(current_point, violated_idx)
            F_repair = np.sum(J_violated, axis=0)
            norm_repair = np.linalg.norm(F_repair)
            if norm_repair > 1e-9:
                F_repair /= norm_repair

        cv = abs(self.constraint_handler.unfeasible_compute(current_point))
        w_repair = np.tanh(cv * 0.5)
        w_search = 1.0 - w_repair
        if not violated_idx:
            w_search, w_repair = 1.0, 0.0

        blended_dir = (w_search * F_search) + (w_repair * F_repair)
        norm_blend = np.linalg.norm(blended_dir)
        if norm_blend < 1e-9:
            blended_dir = F_search + np.random.normal(0, 0.1, size=current_point.shape)
            blended_dir /= np.linalg.norm(blended_dir)

        alpha = 0.5 * self.global_step_factor * (1 + 2.0 * np.tanh(cv))
        return self.encoder.clip_and_round(current_point + alpha * blended_dir)

    def _manifold_correction(self, candidate_point):
        repaired = self.constraint_handler.find_feasible_point(candidate_point, eps=self.tolerance)
        return repaired if repaired is not None else candidate_point
