import numpy as np

from EvolutionaryOptimizer import EvolutionaryOptimizer


def get_cec_g12_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [0, 10]} for i in range(1, 4)}
    constraints = []
    # 生成所有p、q、r ∈ {1,..,9}的组合，共9^3=729个约束
    for p in range(1, 10):
        for q in range(1, 10):
            for r in range(1, 10):
                # 每个约束：(x1-p)^2 + (x2-q)^2 + (x3-r)^2 ≤ 0.0625
                def create_constraint(p_val, q_val, r_val):
                    return lambda x: -(
                                (x['x1'] - p_val) ** 2 + (x['x2'] - q_val) ** 2 + (x['x3'] - r_val) ** 2 - 0.0625)

                constraints.append({'type': 'ineq', 'fun': create_constraint(p, q, r)})
    return {"name": "g12", "latent_dim": 3, "factors": factors, "constraints": constraints}

problem = get_cec_g12_problem()
optimizer = EvolutionaryOptimizer(
            factors=problem['factors'],
            constraints=problem['constraints'],
            max_generations=1000,
            sample_num=10,
            isDraw=False
        )
repair = optimizer.constraint_handler.repair_infeasible_solution(
    np.random.uniform(low=0, high=1, size=3),
    [],
    1e-6,
)
print(optimizer.constraint_handler.cal_val(repair))
