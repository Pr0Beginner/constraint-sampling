import os
import numpy as np
import pandas as pd
import casadi as ca

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 导入刚才写的 Baseline 类
from BaseLine import LHDRejectionSampler


# ================= 补充 E1 到 E5 测试问题 =================

def E1_problem():
    factors = {'x1': {'type': 'cont', 'bounds': [0, 1]}, 'x2': {'type': 'cont', 'bounds': [0, 1]}}
    constraints = [{'type': 'ineq', 'fun': lambda x: -(1 - (x['x1'] - 1) ** 2 - (x['x2'] - 1) ** 2)}]
    return {"name": "E1", "latent_dim": 2, "factors": factors, "constraints": constraints}


def E2_problem():
    factors = {
        'x1': {'type': 'cont', 'bounds': [0, 1]},
        'x2': {'type': 'cont', 'bounds': [0, 1]}
    }
    constraints = [
        {
            'type': 'ineq',
            'fun': lambda x: -ca.fmin(ca.fmax(0.5 - x['x1'], 0.5 - x['x2']), x['x1']**2 + x['x2']**2 - 0.25)
        }
    ]
    return {
        "name": "E2",
        "latent_dim": 2,
        "factors": factors,
        "constraints": constraints
    }


def E3_problem():
    factors = {
        'x1': {'type': 'cont', 'bounds': [78, 102]}, 'x2': {'type': 'cont', 'bounds': [33, 45]},
        'x3': {'type': 'cont', 'bounds': [27, 45]}, 'x4': {'type': 'cont', 'bounds': [27, 45]},
        'x5': {'type': 'cont', 'bounds': [27, 45]},
    }
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(
                    85.334407 + 0.0056858 * x['x2'] * x['x5'] + 0.0006262 * x['x1'] * x['x4'] - 0.0022053 * x['x3'] * x[
                'x5'] - 92)},
        {'type': 'ineq', 'fun': lambda x: -(
                    -85.334407 - 0.0056858 * x['x2'] * x['x5'] - 0.0006262 * x['x1'] * x['x4'] + 0.0022053 * x['x5'] *
                    x['x3'])},
        {'type': 'ineq', 'fun': lambda x: -(
                    80.51249 + 0.0071317 * x['x2'] * x['x5'] + 0.0029955 * x['x1'] * x['x2'] + 0.0021813 * x[
                'x3'] ** 2 - 110)},
        {'type': 'ineq', 'fun': lambda x: -(
                    -80.51249 - 0.0071317 * x['x2'] * x['x5'] - 0.0029955 * x['x1'] * x['x2'] - 0.0021813 * x[
                'x3'] ** 2 + 90)},
        {'type': 'ineq', 'fun': lambda x: -(
                    9.300961 + 0.0047026 * x['x3'] * x['x5'] + 0.0012547 * x['x1'] * x['x3'] + 0.0019085 * x['x3'] * x[
                'x4'] - 25)},
        {'type': 'ineq', 'fun': lambda x: -(
                    -9.300961 - 0.0047026 * x['x3'] * x['x5'] - 0.0012547 * x['x1'] * x['x3'] - 0.0019085 * x['x3'] * x[
                'x4'] + 20)}
    ]
    return {"name": "E3", "latent_dim": 5, "factors": factors, "constraints": constraints}


def E4_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [-10, 10]} for i in range(1, 8)}
    constraints = [
        {'type': 'ineq',
         'fun': lambda x: -(-127 + 2 * x['x1'] ** 2 + 3 * x['x2'] ** 4 + x['x3'] + 4 * x['x4'] ** 2 + 5 * x['x5'])},
        {'type': 'ineq', 'fun': lambda x: -(-282 + 7 * x['x1'] + 3 * x['x2'] + 10 * x['x3'] ** 2 + x['x4'] - x['x5'])},
        {'type': 'ineq', 'fun': lambda x: -(-196 + 23 * x['x1'] + x['x2'] + 6 * x['x6'] - 8 * x['x7'])},
        {'type': 'ineq', 'fun': lambda x: -(
                    4 * x['x1'] ** 2 + x['x2'] ** 2 - 3 * x['x1'] * x['x2'] + 2 * x['x3'] ** 2 + 5 * x['x6'] - 11 * x[
                'x7'])}
    ]
    return {"name": "E4", "latent_dim": 7, "factors": factors, "constraints": constraints}


def E5_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [0, 1]} for i in range(1, 10)}
    for i in range(10, 13):
        factors[f'x{i}'] = {'type': 'cont', 'bounds': [0, 100]}
    factors['x13'] = {'type': 'cont', 'bounds': [0, 1]}
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(2 * x['x1'] + 2 * x['x2'] + x['x10'] + x['x11'] - 10)},
        {'type': 'ineq', 'fun': lambda x: -(2 * x['x1'] + 2 * x['x3'] + x['x10'] + x['x12'] - 10)},
        {'type': 'ineq', 'fun': lambda x: -(2 * x['x2'] + 2 * x['x3'] + x['x11'] + x['x12'] - 10)},
        {'type': 'ineq', 'fun': lambda x: -(-8 * x['x1'] + x['x10'])},
        {'type': 'ineq', 'fun': lambda x: -(-8 * x['x2'] + x['x11'])},
        {'type': 'ineq', 'fun': lambda x: -(-8 * x['x3'] + x['x12'])},
        {'type': 'ineq', 'fun': lambda x: -(-2 * x['x4'] - x['x5'] + x['x10'])},
        {'type': 'ineq', 'fun': lambda x: -(-2 * x['x6'] - x['x7'] + x['x11'])},
        {'type': 'ineq', 'fun': lambda x: -(-2 * x['x8'] - x['x9'] + x['x12'])}
    ]
    return {"name": "E5", "latent_dim": 13, "factors": factors, "constraints": constraints}


if __name__ == "__main__":
    test_problems = [
        E1_problem(), E2_problem(), E3_problem(), E4_problem(), E5_problem()
    ]

    sample_nums = [20, 50]
    num_runs = 10  # Baseline 运行次数

    # 打印表头（把 方差 改为 标准差）
    print(f"{'用例':<5} | {'目标采样':<8} | {'实际平均采样':<12} | {'准则值均值':<15} | {'准则值标准差'}")
    print("-" * 65)

    for sample_num in sample_nums:
        for problem in test_problems:
            uniformity_list = []
            final_sample_count_list = []

            for run_idx in range(num_runs):
                sampler = LHDRejectionSampler(
                    factors=problem['factors'],
                    constraints=problem['constraints'],
                    sample_num=sample_num,
                    max_evaluations=2000000  # 最大允许盲抽200万次
                )

                final_samples = sampler.optimize()

                final_sample_count = len(final_samples)
                final_sample_count_list.append(final_sample_count)

                if final_sample_count > 0:
                    uniformity_list.append(sampler.best_val)
                else:
                    uniformity_list.append(np.nan)

            valid_uniformity = [u for u in uniformity_list if not np.isnan(u)]
            avg_final_sample_count = np.mean(final_sample_count_list)

            if len(valid_uniformity) > 0:
                avg_uniformity = np.mean(valid_uniformity)
                # ====================== 这里改为 标准差 ======================
                std_uniformity = np.std(valid_uniformity)
            else:
                avg_uniformity = float('inf')
                std_uniformity = 0.0

            # 输出：方差 → 标准差
            print(f"{problem['name']:<7} | {sample_num:<10} | {avg_final_sample_count:<18.2f} | {avg_uniformity:<19.6f} | {std_uniformity:.6f}")
