import numpy as np
import casadi as ca


class ConstraintHandler:
    def __init__(self, encoder, constraints, tolerance=1e-8):
        self.encoder = encoder
        self.constraints = constraints
        self.tolerance = tolerance
        self.x_sym = ca.SX.sym("x", self.encoder.total_dims)

        self._ineq_idx = np.array(
            [i for i, con in enumerate(self.constraints) if con["type"] == "ineq"], dtype=int
        )
        self._eq_idx = np.array(
            [i for i, con in enumerate(self.constraints) if con["type"] == "eq"], dtype=int
        )
        self._has_constraints = len(self.constraints) > 0

        self._base_lbx, self._base_ubx = self._build_base_bounds()
        self._build_fast_evaluators()
        self._solver_dict = self._create_solver(tolerance=tolerance)

    def _build_base_bounds(self):
        lbx = np.empty(self.encoder.total_dims, dtype=float)
        ubx = np.empty(self.encoder.total_dims, dtype=float)
        for i in range(self.encoder.total_dims):
            info = self.encoder.factor_info_by_index[i]["info"]
            if info["type"] == "cont":
                lbx[i] = 0.0
                ubx[i] = 1.0
            elif info["type"] == "int":
                lbx[i] = info["bounds"][0]
                ubx[i] = info["bounds"][1]
            else:
                lbx[i] = 0.0
                ubx[i] = len(info["options"]) - 1
        return lbx, ubx

    def _build_fast_evaluators(self):
        raw_g_list = []
        decoded_sym = {}

        for name, idx in self.encoder.cont_pos.items():
            low, high = self.encoder.factors[name]["bounds"]
            decoded_sym[name] = self.x_sym[idx] * (high - low) + low

        for name, info in self.encoder.disc_pos.items():
            decoded_sym[name] = self.x_sym[info["index"]]

        for con in self.constraints:
            raw_g_list.append(con["fun"](decoded_sym))

        if raw_g_list:
            self.raw_g_sym = ca.vertcat(*raw_g_list)
            self.eval_con = ca.Function("eval_con", [self.x_sym], [self.raw_g_sym])
            self.raw_jac_sym = ca.jacobian(self.raw_g_sym, self.x_sym)
            self.eval_jac = ca.Function("eval_jac", [self.x_sym], [self.raw_jac_sym])
        else:
            self.raw_g_sym = ca.SX([])
            self.eval_con = None
            self.eval_jac = None

    def _create_solver(
        self,
        objective_builder=None,
        extra_constraints_builder=None,
        num_extra_vars=0,
        tolerance=1e-8,
    ):
        solver_options = {
            "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 1000, "tol": tolerance},
            "print_time": 0,
        }

        c_sym = ca.SX.sym("c", self.encoder.total_dims)
        decision_vars = self.x_sym
        if num_extra_vars > 0:
            extra_vars_sym = ca.SX.sym("extra", num_extra_vars)
            decision_vars = ca.vertcat(self.x_sym, extra_vars_sym)

        g_ineq_list = []
        g_eq_list = []
        if self.eval_con is not None:
            raw_expr = self.eval_con(self.x_sym)
            if self._ineq_idx.size > 0:
                g_ineq_list.append(-raw_expr[self._ineq_idx.tolist()])
            if self._eq_idx.size > 0:
                g_eq_list.append(raw_expr[self._eq_idx.tolist()])

        if extra_constraints_builder:
            extra_ineq, extra_eq = extra_constraints_builder(decision_vars)
            g_ineq_list.extend(extra_ineq)
            g_eq_list.extend(extra_eq)

        g_ineq = ca.vertcat(*g_ineq_list) if g_ineq_list else ca.SX([])
        g_eq = ca.vertcat(*g_eq_list) if g_eq_list else ca.SX([])
        g = ca.vertcat(g_ineq, g_eq)

        num_ineq_constraints = g_ineq.shape[0]
        num_eq_constraints = g_eq.shape[0]

        if objective_builder:
            objective = objective_builder(decision_vars, c_sym)
        else:
            objective = ca.sumsqr(self.x_sym - c_sym)

        nlp_prob = {"f": objective, "x": decision_vars, "g": g, "p": c_sym}
        solver = ca.nlpsol("solver", "ipopt", nlp_prob, solver_options)

        lbg = np.concatenate(
            [-np.inf * np.ones(num_ineq_constraints), np.zeros(num_eq_constraints)]
        )
        ubg = np.concatenate([np.zeros(num_ineq_constraints), np.zeros(num_eq_constraints)])

        return {
            "solver": solver,
            "lbx": self._base_lbx.copy(),
            "ubx": self._base_ubx.copy(),
            "lbg": lbg,
            "ubg": ubg,
        }

    def _eval_constraint_values(self, sample):
        if not self._has_constraints:
            return np.empty(0, dtype=float)
        return np.asarray(self.eval_con(sample).full(), dtype=float).reshape(-1)

    def evaluate_feasibility_and_cv(self, sample):
        if not self._has_constraints:
            return True, 0.0

        vals = self._eval_constraint_values(sample)

        feasible = True
        if self._ineq_idx.size > 0 and np.any(vals[self._ineq_idx] < -self.tolerance):
            feasible = False
        if self._eq_idx.size > 0 and np.any(np.abs(vals[self._eq_idx]) > self.tolerance):
            feasible = False

        cv = 0.0
        if self._ineq_idx.size > 0:
            ineq_vals = vals[self._ineq_idx]
            cv += float(np.sum(np.maximum(0.0, -ineq_vals)))
        if self._eq_idx.size > 0:
            cv += float(np.sum(np.abs(vals[self._eq_idx])))

        return feasible, cv

    def find_feasible_point(self, initial_guess, eps=1e-6):
        try:
            sol = self._solver_dict["solver"](
                p=initial_guess,
                lbx=self._solver_dict["lbx"],
                ubx=self._solver_dict["ubx"],
                lbg=self._solver_dict["lbg"],
                ubg=self._solver_dict["ubg"],
                x0=initial_guess,
            )
            final_point_rounded = self.encoder.clip_and_round(sol["x"].full().flatten())
            if self.judge_feasible(final_point_rounded):
                return final_point_rounded
            return None
        except Exception:
            return None

    def find_diverse_feasible_point(self, initial_guess, existing_points, eps=1e-6):
        if not existing_points:
            return self.find_feasible_point(initial_guess, eps)

        existing_points_arr = np.asarray(existing_points)

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
            num_extra_vars=1,
        )

        lbx_diverse = np.append(diverse_solver_dict["lbx"], 0.0)
        ubx_diverse = np.append(diverse_solver_dict["ubx"], np.inf)

        try:
            initial_guess_diverse = np.append(initial_guess, 0.0)
            sol = diverse_solver_dict["solver"](
                p=initial_guess,
                lbx=lbx_diverse,
                ubx=ubx_diverse,
                lbg=diverse_solver_dict["lbg"],
                ubg=diverse_solver_dict["ubg"],
                x0=initial_guess_diverse,
            )
            final_point_rounded = self.encoder.clip_and_round(sol["x"].full().flatten()[:-1])
            if self.judge_feasible(final_point_rounded):
                return final_point_rounded
            return None
        except Exception:
            return self.find_feasible_point(initial_guess, eps)

    def repair_infeasible_solution(self, candidate_point, existing_points=None, eps=1e-6, max_attempts=5):
        if existing_points is None:
            existing_points = []

        if self.judge_feasible(candidate_point):
            return candidate_point

        repaired_point = self.find_feasible_point(candidate_point, eps)
        if repaired_point is not None:
            return repaired_point

        if existing_points:
            repaired_point = self.find_diverse_feasible_point(candidate_point, existing_points, eps)
            if repaired_point is not None:
                return repaired_point

        for _ in range(max_attempts):
            perturbation = np.random.normal(0, 0.1, self.encoder.total_dims)
            perturbed_point = self.encoder.clip_and_round(candidate_point + perturbation)
            repaired_point = self.find_feasible_point(perturbed_point, eps)
            if repaired_point is not None:
                return repaired_point
            if existing_points:
                repaired_point = self.find_diverse_feasible_point(perturbed_point, existing_points, eps)
                if repaired_point is not None:
                    return repaired_point
        return None

    def judge_feasible(self, sample):
        if not self._has_constraints:
            return True
        vals = self._eval_constraint_values(sample)
        if self._ineq_idx.size > 0 and np.any(vals[self._ineq_idx] < -self.tolerance):
            return False
        if self._eq_idx.size > 0 and np.any(np.abs(vals[self._eq_idx]) > self.tolerance):
            return False
        return True

    def unfeasible_compute(self, sample):
        if not self._has_constraints:
            return 0.0
        vals = self._eval_constraint_values(sample)
        cv = 0.0
        if self._ineq_idx.size > 0:
            cv += float(np.sum(np.maximum(0.0, -vals[self._ineq_idx])))
        if self._eq_idx.size > 0:
            cv += float(np.sum(np.abs(vals[self._eq_idx])))
        return cv

    def cal_val(self, sin_sample):
        return self.unfeasible_compute(sin_sample)

    @staticmethod
    def is_feasible(sample, encoder, constraints, eps=1e-6):
        decoded = encoder.decode(sample)
        for con in constraints:
            try:
                val = con["fun"](decoded)
                if hasattr(val, "full"):
                    val = val.full().item()
                if con["type"] == "ineq" and val < -eps:
                    return False
                if con["type"] == "eq" and abs(val) > eps:
                    return False
            except Exception:
                return False
        return True

    def project_to_equality(self, sample, encoder, equality_constraints, eps=1e-6):
        if not equality_constraints:
            return sample
        temp_handler = ConstraintHandler(encoder, equality_constraints)
        return temp_handler.find_feasible_point(sample, eps)

    @staticmethod
    def check_pure_int_cat_constraints(encoder, constraints):
        cont_vars = {
            var["name"]
            for var in encoder.factor_info_by_index.values()
            if var["info"]["type"] == "cont"
        }
        results = []
        for constraint in constraints:
            fun_str = str(constraint["fun"])
            results.append(not any(var in fun_str for var in cont_vars))
        return results
