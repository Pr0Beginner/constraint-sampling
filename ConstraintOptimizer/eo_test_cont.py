import os
import numpy as np
import matplotlib.pyplot as plt
import casadi
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from matplotlib import font_manager
from OldEvoluationaryOptimizer import EvolutionaryOptimizer
from utils.utils import MyUtils

try:
    font_path = "C:/Windows/Fonts/msyh.ttc"
    font = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except FileNotFoundError:
    print("未找到微软雅黑字体，将使用默认字体。")

def get_cec_g01_problem():
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
    return {"name": "g01", "latent_dim": 13, "factors": factors, "constraints": constraints}

def get_cec_g02_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [0, 10]} for i in range(1, 21)}

    # 直接定义两个独立的约束函数，避免返回序列
    def constraint1(x):
        n = 20
        x_vec = [x[f'x{i}'] for i in range(1, n + 1)]
        product = 1.0
        for val in x_vec:
            product *= val
        return product - 0.75  # 约束1: prod(x) >= 0.75

    def constraint2(x):
        n = 20
        x_vec = [x[f'x{i}'] for i in range(1, n + 1)]
        total_sum = sum(x_vec)
        return 7.5 * n - total_sum  # 约束2: sum(x) <= 7.5*n

    # 分别使用独立的约束函数，确保每个函数返回标量
    constraints = [
        {'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2}
    ]

    return {"name": "g02", "latent_dim": 20, "factors": factors, "constraints": constraints}

def get_cec_g03_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [0, 1]} for i in range(1, 11)}
    constraints = [
        {'type': 'eq', 'fun': lambda x: casadi.sumsqr(casadi.vertcat(*[x[f'x{i}'] for i in range(1, 11)])) - 1}
    ]
    return {"name": "g03", "latent_dim": 10, "factors": factors, "constraints": constraints}

def get_cec_g04_problem():
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
    return {"name": "g04", "latent_dim": 5, "factors": factors, "constraints": constraints}

def get_cec_g05_problem():
    factors = {
        'x1': {'type': 'cont', 'bounds': [0, 1200]}, 'x2': {'type': 'cont', 'bounds': [0, 1200]},
        'x3': {'type': 'cont', 'bounds': [-0.55, 0.55]}, 'x4': {'type': 'cont', 'bounds': [-0.55, 0.55]},
    }
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(-x['x4'] + x['x3'] - 0.55)},
        {'type': 'ineq', 'fun': lambda x: -(-x['x3'] + x['x4'] - 0.55)},
        {'type': 'eq',
         'fun': lambda x: 1000 * casadi.sin(-x['x3'] - 0.25) + 1000 * casadi.sin(-x['x4'] - 0.25) + 894.8 - x['x1']},
        {'type': 'eq',
         'fun': lambda x: 1000 * casadi.sin(x['x3'] - 0.25) + 1000 * casadi.sin(x['x3'] - x['x4'] - 0.25) + 894.8 - x['x2']},
        {'type': 'eq',
         'fun': lambda x: 1000 * casadi.sin(x['x4'] - 0.25) + 1000 * casadi.sin(x['x4'] - x['x3'] - 0.25) + 1294.8}
    ]
    return {"name": "g05", "latent_dim": 4, "factors": factors, "constraints": constraints}

def get_cec_g06_problem():
    factors = {'x1': {'type': 'cont', 'bounds': [13, 100]}, 'x2': {'type': 'cont', 'bounds': [0, 100]}}
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(-(x['x1'] - 5) ** 2 - (x['x2'] - 5) ** 2 + 100)},
        {'type': 'ineq', 'fun': lambda x: -((x['x1'] - 6) ** 2 + (x['x2'] - 5) ** 2 - 82.81)}
    ]
    return {"name": "g06", "latent_dim": 2, "factors": factors, "constraints": constraints}

def get_cec_g07_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [-10, 10]} for i in range(1, 11)}
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(-105 + 4 * x['x1'] + 5 * x['x2'] - 3 * x['x7'] + 9 * x['x8'])},
        {'type': 'ineq', 'fun': lambda x: -(10 * x['x1'] - 8 * x['x2'] - 17 * x['x7'] + 2 * x['x8'])},
        {'type': 'ineq', 'fun': lambda x: -(-8 * x['x1'] + 2 * x['x2'] + 5 * x['x9'] - 2 * x['x10'] - 12)},
        {'type': 'ineq',
         'fun': lambda x: -(3 * (x['x1'] - 2) ** 2 + 4 * (x['x2'] - 3) ** 2 + 2 * x['x3'] ** 2 - 7 * x['x4'] - 120)},
        {'type': 'ineq', 'fun': lambda x: -(5 * x['x1'] ** 2 + 8 * x['x2'] + (x['x3'] - 6) ** 2 - 2 * x['x4'] - 40)},
        {'type': 'ineq', 'fun': lambda x: -(
                    x['x1'] ** 2 + 2 * (x['x2'] - 2) ** 2 - 2 * x['x1'] * x['x2'] + 14 * x['x5'] - 6 * x['x6'])},
        {'type': 'ineq',
         'fun': lambda x: -(0.5 * (x['x1'] - 8) ** 2 + 2 * (x['x2'] - 4) ** 2 + 3 * x['x5'] ** 2 - x['x6'] - 30)},
        {'type': 'ineq', 'fun': lambda x: -(-3 * x['x1'] + 6 * x['x2'] + 12 * (x['x9'] - 8) ** 2 - 7 * x['x10'])}
    ]
    return {"name": "g07", "latent_dim": 10, "factors": factors, "constraints": constraints}

def get_cec_g08_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [0, 10]} for i in range(1, 3)}
    constraints = [
        # 正确：使用字典键访问变量（与factors定义的x1、x2对应）
        {'type': 'ineq', 'fun': lambda x: -(x['x1']** 2 - x['x2'] + 1)},
        {'type': 'ineq', 'fun': lambda x: -(1 - x['x1'] + (x['x2'] - 4) **2)}
    ]
    return {"name": "g08", "latent_dim": 2, "factors": factors, "constraints": constraints}

def get_cec_g09_problem():
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
    return {"name": "g09", "latent_dim": 7, "factors": factors, "constraints": constraints}

def get_cec_g10_problem():
    factors = {'x1': {'type': 'cont', 'bounds': [100, 10000]}, 'x2': {'type': 'cont', 'bounds': [1000, 10000]},
               'x3': {'type': 'cont', 'bounds': [1000, 10000]}}
    for i in range(4, 9):
        factors[f'x{i}'] = {'type': 'cont', 'bounds': [10, 1000]}
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(-1 + 0.0025 * (x['x4'] + x['x6']))},
        {'type': 'ineq', 'fun': lambda x: -(-1 + 0.0025 * (x['x5'] + x['x7'] - x['x4']))},
        {'type': 'ineq', 'fun': lambda x: -(-1 + 0.01 * (x['x8'] - x['x5']))},
        {'type': 'ineq', 'fun': lambda x: -(-x['x1'] * x['x6'] + 833.33252 * x['x4'] + 100 * x['x1'] - 83333.333)},
        {'type': 'ineq', 'fun': lambda x: -(-x['x2'] * x['x7'] + 1250 * x['x5'] + x['x2'] * x['x4'] - 1250 * x['x4'])},
        {'type': 'ineq', 'fun': lambda x: -(-x['x3'] * x['x8'] + 1250000 + x['x3'] * x['x5'] - 2500 * x['x5'])}
    ]
    return {"name": "g10", "latent_dim": 8, "factors": factors, "constraints": constraints}

def get_cec_g11_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [-1, 1]} for i in range(1, 3)}
    constraints = [{'type': 'eq', 'fun': lambda x: x['x2'] - x['x1'] ** 2}]
    return {"name": "g11", "latent_dim": 2, "factors": factors, "constraints": constraints}

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

def get_cec_g13_problem():
    factors = {'x1': {'type': 'cont', 'bounds': [-2.3, 2.3]}, 'x2': {'type': 'cont', 'bounds': [-2.3, 2.3]},
               'x3': {'type': 'cont', 'bounds': [-3.2, 3.2]}, 'x4': {'type': 'cont', 'bounds': [-3.2, 3.2]},
               'x5': {'type': 'cont', 'bounds': [-3.2, 3.2]}}
    constraints = [
        {'type': 'eq', 'fun': lambda x: x['x1'] ** 2 + x['x2'] ** 2 + x['x3'] ** 2 + x['x4'] ** 2 + x['x5'] ** 2 - 10},
        {'type': 'eq', 'fun': lambda x: x['x2'] * x['x3'] - 5 * x['x4'] * x['x5']},
        {'type': 'eq', 'fun': lambda x: x['x1'] ** 3 + x['x2'] ** 3 + 1}
    ]
    return {"name": "g13", "latent_dim": 5, "factors": factors, "constraints": constraints}

def get_cec_g14_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [1e-6, 10]} for i in range(1, 11)}
    constraints = [
        {'type': 'eq', 'fun': lambda x: x['x1'] + 2 * x['x2'] + 2 * x['x3'] + x['x6'] + x['x10'] - 2},
        {'type': 'eq', 'fun': lambda x: x['x4'] + 2 * x['x5'] + x['x6'] + x['x7'] - 1},
        {'type': 'eq', 'fun': lambda x: x['x3'] + x['x7'] + x['x8'] + 2 * x['x9'] + x['x10'] - 1}
    ]
    return {"name": "g14", "latent_dim": 10, "factors": factors, "constraints": constraints}

def get_cec_g15_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [0, 10]} for i in range(1, 4)}
    constraints = [
        {'type': 'eq', 'fun': lambda x: x['x1'] ** 2 + x['x2'] ** 2 + x['x3'] ** 2 - 25},
        {'type': 'eq', 'fun': lambda x: 8 * x['x1'] + 14 * x['x2'] + 7 * x['x3'] - 56}
    ]
    return {"name": "g15", "latent_dim": 3, "factors": factors, "constraints": constraints}

def get_cec_g16_problem():
    factors = {'x1': {'type': 'cont', 'bounds': [704.4148, 906.3855]}, 'x2': {'type': 'cont', 'bounds': [68.6, 288.88]},
               'x3': {'type': 'cont', 'bounds': [0, 134.75]}, 'x4': {'type': 'cont', 'bounds': [193, 287.0966]},
               'x5': {'type': 'cont', 'bounds': [25, 84.1988]}}

    def complex_constraints(x):
        y1 = x['x2'] + x['x3'] + 41.6
        c1 = 0.024 * x['x4'] - 4.62
        y2 = 12.5 / c1 + 12
        c2 = 0.0003535 * x['x1'] ** 2 + 0.5311 * x['x1'] + 0.08705 * y2 * x['x1']
        c3 = 0.052 * x['x1'] + 78 + 0.002377 * y2 * x['x1']
        y3 = c2 / c3
        y4 = 19 * y3
        c4 = 0.04782 * (x['x1'] - y3) + (0.1956 * (x['x1'] - y3) ** 2) / x['x2'] + 0.6376 * y4 + 1.594 * y3
        c5 = 100 * x['x2']
        c6 = x['x1'] - y3 - y4
        c7 = 0.950 - c4 / c5
        y5 = c6 * c7
        y6 = x['x1'] - y5 - y4 - y3
        c8 = (y5 + y4) * 0.995
        y7 = c8 / y1
        y8 = c8 / 3798
        c9 = y7 - (0.0663 * y7) / y8 - 0.3153
        y9 = 96.82 / c9 + 0.321 * y1
        y10 = 1.29 * y5 + 1.258 * y4 + 2.29 * y3 + 1.71 * y6
        y11 = 1.71 * x['x1'] - 0.452 * y4 + 0.580 * y3
        c10 = 12.3 / 752.3
        c11 = (1.75 * y2) * (0.995 * x['x1'])
        c12 = 0.995 * y10 + 1998
        y12 = c10 * x['x1'] + c11 / c12
        y13 = c12 - 1.75 * y2
        c17 = y9 + x['x5']
        y14 = 3623 + 64.4 * x['x2'] + 58.4 * x['x3'] + 146312 / c17
        c13 = 0.995 * y10 + 60.8 * x['x2'] + 48 * x['x4'] - 0.1121 * y14 - 5095
        y15 = y13 / c13
        y16 = 148000 - 331000 * y15 + 40 * y13 - 61 * y15 * y13
        c14 = 2324 * y10 - 28740000 * y2
        y17 = 14130000 - 1328 * y10 - 531 * y11 + c14 / c12

        g = [
            (0.28 / 0.72) * y5 - y4, x['x3'] - 1.5 * x['x2'], 3496 * y2 / c12 - 21,
            110.6 + y1 - (62212 / c17), 213.1 - y1, y1 - 405.23,
            17.505 - y2, y2 - 1053.6667, 11.275 - y3, y3 - 35.03,
            214.228 - y4, y4 - 665.585, 7.458 - y5, y5 - 584.463,
            0.961 - y6, y6 - 265.916, 1.612 - y7, y7 - 7.046,
            0.146 - y8, y8 - 0.222, 107.99 - y9, y9 - 273.366,
            922.693 - y10, y10 - 1286.105, 926.832 - y11, y11 - 1444.046,
            18.766 - y12, y12 - 537.141, 1072.163 - y13, y13 - 3247.039,
            8961.448 - y14, y14 - 26844.086, 0.063 - y15, y15 - 0.386,
            71084.33 - y16, -140000 + y16, 2802713 - y17, y17 - 12146108
        ]
        return g

    constraints = [{'type': 'ineq', 'fun': lambda x, i=i: -complex_constraints(x)[i]} for i in
                   range(len(complex_constraints({f'x1': 1, 'x2': 1, 'x3': 1, 'x4': 1, 'x5': 1})))]
    return {"name": "g16", "latent_dim": 5, "factors": factors, "constraints": constraints}

def get_cec_g17_problem():
    factors = {
        'x1': {'type': 'cont', 'bounds': [0, 400]}, 'x2': {'type': 'cont', 'bounds': [0, 1000]},
        'x3': {'type': 'cont', 'bounds': [340, 420]}, 'x4': {'type': 'cont', 'bounds': [340, 420]},
        'x5': {'type': 'cont', 'bounds': [-1000, 1000]}, 'x6': {'type': 'cont', 'bounds': [0, 0.5236]}
    }
    constraints = [
        {'type': 'eq', 'fun': lambda x: -x['x1'] + 300 - (x['x3'] * x['x4'] / 131.078) * casadi.cos(1.48477 - x['x6']) + (
                    0.90798 * x['x3'] ** 2 / 131.078) * casadi.cos(1.47588)},
        {'type': 'eq', 'fun': lambda x: -x['x2'] - (x['x3'] * x['x4'] / 131.078) * casadi.cos(1.48477 + x['x6']) + (
                    0.90798 * x['x4'] ** 2 / 131.078) * casadi.cos(1.47588)},
        {'type': 'eq', 'fun': lambda x: -x['x5'] - (x['x3'] * x['x4'] / 131.078) * casadi.sin(1.48477 + x['x6']) + (
                    0.90798 * x['x4'] ** 2 / 131.078) * casadi.sin(1.47588)},
        {'type': 'eq', 'fun': lambda x: 200 - (x['x3'] * x['x4'] / 131.078) * casadi.sin(1.48477 - x['x6']) + (
                    0.90798 * x['x3'] ** 2 / 131.078) * casadi.sin(1.47588)}
    ]
    return {"name": "g17", "latent_dim": 6, "factors": factors, "constraints": constraints}

def get_cec_g18_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [-10, 10]} for i in range(1, 9)}
    factors['x9'] = {'type': 'cont', 'bounds': [0, 20]}
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(x['x3'] ** 2 + x['x4'] ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -(x['x9'] ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -(x['x5'] ** 2 + x['x6'] ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -(x['x1'] ** 2 + (x['x2'] - x['x9']) ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -((x['x1'] - x['x5']) ** 2 + (x['x2'] - x['x6']) ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -((x['x1'] - x['x7']) ** 2 + (x['x2'] - x['x8']) ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -((x['x3'] - x['x5']) ** 2 + (x['x4'] - x['x6']) ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -((x['x3'] - x['x7']) ** 2 + (x['x4'] - x['x8']) ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -(x['x7'] ** 2 + (x['x8'] - x['x9']) ** 2 - 1)},
        {'type': 'ineq', 'fun': lambda x: -(x['x2'] * x['x3'] - x['x1'] * x['x4'])},
        {'type': 'ineq', 'fun': lambda x: -(-x['x3'] * x['x9'])},
        {'type': 'ineq', 'fun': lambda x: -(x['x5'] * x['x9'])},
        {'type': 'ineq', 'fun': lambda x: -(x['x6'] * x['x7'] - x['x5'] * x['x8'])}
    ]
    return {"name": "g18", "latent_dim": 9, "factors": factors, "constraints": constraints}

def get_cec_g19_problem():
    factors = {f'x{i}': {'type': 'cont', 'bounds': [0, 10]} for i in range(1, 16)}
    # CEC2006 g19标准参数矩阵
    C = np.array([
        [30, -20, -10, 32, -10],
        [-20, 39, -6, -31, 32],
        [-10, -6, 10, -6, -10],
        [32, -31, -6, 39, -20],
        [-10, 32, -10, -20, 30]
    ])
    d = np.array([4, 8, 10, 6, 2])
    e = np.array([-15, -27, -36, -18, -12])
    a = np.array([
        [-16, 2, 0, 1, 0],
        [0, -2, 0, 0.4, 2],
        [-3.5, 0, 2, 0, 0],
        [0, -2, 0, -4, -1],
        [0, -9, -2, 1, -2.8],
        [2, 0, -4, 0, 0],
        [-1, -1, -1, -1, -1],
        [-1, -2, -3, -2, -1],
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1]
    ])

    constraints = []
    # 生成5个非线性不等式约束
    for j in range(5):
        def create_constraint(j_val):
            return lambda x: -(
                    -2 * casadi.sum1(casadi.vertcat(*[C[i, j_val] * x[f'x{11 + i}'] for i in range(5)]))
                    - 3 * d[j_val] * x[f'x{11 + j_val}'] ** 2
                    - e[j_val]
                    + casadi.sum1(casadi.vertcat(*[a[i, j_val] * x[f'x{i + 1}'] for i in range(10)]))
            )
        constraints.append({'type': 'ineq', 'fun': create_constraint(j)})

    # 目标函数（注：原代码中优化器会最小化该函数，与CEC2006定义一致）
    def objective(x):
        x11_x15 = np.array([x[f'x{i}'] for i in range(11, 16)])
        quadratic_term = np.dot(np.dot(x11_x15.T, C), x11_x15)
        cubic_term = 2 * np.sum(d * x11_x15 ** 3)
        linear_term = np.sum([1 * x[f'x{i + 1}'] for i in range(10)])  # b_i=1（CEC2006标准）
        return quadratic_term + cubic_term - linear_term

    return {
        "name": "g19",
        "latent_dim": 15,
        "factors": factors,
        "constraints": constraints,
        "objective": objective  # 显式指定目标函数（部分优化器需单独传入）
    }

def get_cec_g21_problem():
    factors = {
        'x1': {'type': 'cont', 'bounds': [0, 1000]}, 'x2': {'type': 'cont', 'bounds': [0, 40]},
        'x3': {'type': 'cont', 'bounds': [0, 40]},
        'x4': {'type': 'cont', 'bounds': [100, 300]}, 'x5': {'type': 'cont', 'bounds': [6.3, 6.7]},
        'x6': {'type': 'cont', 'bounds': [5.9, 6.4]},
        'x7': {'type': 'cont', 'bounds': [4.5, 6.25]}
    }
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(-x['x1'] + 35 * x['x2'] ** 0.6 + 35 * x['x3'] ** 0.6)},
        {'type': 'eq',
         'fun': lambda x: -300 * x['x3'] + 7500 * x['x5'] - 7500 * x['x6'] - 25 * x['x4'] * x['x5'] + 25 * x['x4'] * x[
             'x6'] + x['x3'] * x['x4']},
        {'type': 'eq',
         'fun': lambda x: 100 * x['x2'] + 155.365 * x['x4'] + 2500 * x['x7'] - x['x2'] * x['x4'] - 25 * x['x4'] * x[
             'x7'] - 15536.5},
        {'type': 'eq', 'fun': lambda x: -x['x5'] + casadi.log(-x['x4'] + 900)},
        {'type': 'eq', 'fun': lambda x: -x['x6'] + casadi.log(x['x4'] + 300)},
        {'type': 'eq', 'fun': lambda x: -x['x7'] + casadi.log(-2 * x['x4'] + 700)}
    ]
    return {"name": "g21", "latent_dim": 7, "factors": factors, "constraints": constraints}

def get_cec_g23_problem():
    factors = {
        'x1': {'type': 'cont', 'bounds': [0, 300]}, 'x2': {'type': 'cont', 'bounds': [0, 300]},
        'x3': {'type': 'cont', 'bounds': [0, 100]}, 'x4': {'type': 'cont', 'bounds': [0, 200]},
        'x5': {'type': 'cont', 'bounds': [0, 100]}, 'x6': {'type': 'cont', 'bounds': [0, 300]},
        'x7': {'type': 'cont', 'bounds': [0, 100]}, 'x8': {'type': 'cont', 'bounds': [0, 200]},
        'x9': {'type': 'cont', 'bounds': [0.01, 0.03]}
    }
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(x['x9'] * x['x3'] + 0.02 * x['x6'] - 0.025 * x['x5'])},
        {'type': 'ineq', 'fun': lambda x: -(x['x9'] * x['x4'] + 0.02 * x['x7'] - 0.015 * x['x8'])},
        {'type': 'eq', 'fun': lambda x: x['x1'] + x['x2'] - x['x3'] - x['x4']},
        {'type': 'eq', 'fun': lambda x: 0.03 * x['x1'] + 0.01 * x['x2'] - x['x9'] * (x['x3'] + x['x4'])},
        {'type': 'eq', 'fun': lambda x: x['x3'] + x['x6'] - x['x5']},
        {'type': 'eq', 'fun': lambda x: x['x4'] + x['x7'] - x['x8']}
    ]
    return {"name": "g23", "latent_dim": 9, "factors": factors, "constraints": constraints}

def get_cec_g24_problem():
    factors = {'x1': {'type': 'cont', 'bounds': [0, 3]}, 'x2': {'type': 'cont', 'bounds': [0, 4]}}
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -(-2 * x['x1'] ** 4 + 8 * x['x1'] ** 3 - 8 * x['x1'] ** 2 + x['x2'] - 2)},
        {'type': 'ineq',
         'fun': lambda x: -(-4 * x['x1'] ** 4 + 32 * x['x1'] ** 3 - 88 * x['x1'] ** 2 + 96 * x['x1'] + x['x2'] - 36)}
    ]
    return {"name": "g24", "latent_dim": 2, "factors": factors, "constraints": constraints}

if __name__ == "__main__":
    # 包含所有能求解的
    # all_cec_problems = [
    #     get_cec_g01_problem(), get_cec_g02_problem(), get_cec_g03_problem(), get_cec_g04_problem(),
    #     get_cec_g05_problem(), get_cec_g06_problem(), get_cec_g07_problem(), get_cec_g08_problem(),
    #     get_cec_g09_problem(), get_cec_g10_problem(), get_cec_g11_problem(), get_cec_g12_problem(),
    #     get_cec_g13_problem(), get_cec_g14_problem(), get_cec_g15_problem(), get_cec_g16_problem(),
    #     get_cec_g17_problem(), get_cec_g18_problem(), get_cec_g19_problem(), get_cec_g23_problem(),
    #     get_cec_g24_problem(),
    # ]
    inequality_problems = [
        get_cec_g01_problem(), get_cec_g02_problem(), get_cec_g04_problem(),
        get_cec_g06_problem(), get_cec_g07_problem(), get_cec_g08_problem(),
        get_cec_g09_problem(), get_cec_g10_problem(), get_cec_g16_problem(),
        get_cec_g18_problem(), get_cec_g19_problem(), get_cec_g24_problem(),
    ]
    # inequality_problems = [
    #     get_cec_g02_problem()
    # ]
    for problem in inequality_problems:
        print(f"\n{'=' * 50}")
        print(f"开始处理问题: {problem['name']} (共运行10次)")
        print(f"{'=' * 50}")
        # 初始化存储10次运行结果的数组
        uniformity_list = []  # 均匀性指标
        constraint_violation_list = []  # 约束违反程度
        final_sample_count_list = []  # 最终样本数
        # 对当前问题运行10次优化
        for run_idx in range(0, 1):
            print(f"\n--- 第 {run_idx}/10 次运行 ---")
            # 初始化优化器
            optimizer = EvolutionaryOptimizer(
                factors=problem['factors'],
                constraints=problem['constraints'],
                max_generations=1500,
                sample_num=20,
                isDraw=False
            )
            # 执行优化
            final_samples = optimizer.optimize()
            # 记录当前运行的关键指标
            final_sample_count = len(final_samples)
            uniformity = optimizer.uniformity_evaluator.evaluate(final_samples)
            constraint_violation = np.sum([
                optimizer.constraint_handler.cal_val(sample)
                for sample in optimizer.elite_population
            ])
            # 存储到数组中
            final_sample_count_list.append(final_sample_count)
            uniformity_list.append(uniformity)
            constraint_violation_list.append(constraint_violation)
            # 打印当前运行结果
            print(f"{problem['name']} 第{run_idx}次 - 最终样本数: {final_sample_count}")
            print(f"{problem['name']} 第{run_idx}次 - 均匀性: {uniformity:.6f}")
            print(f"{problem['name']} 第{run_idx}次 - 约束违反程度: {constraint_violation:.6f}")
            # 可选：绘制每次的优化历史（若需要保留）
            # MyUtils.plot_optimization_history(optimizer.history[1:], f"{problem['name']} 第{run_idx}次优化过程")
        # 计算10次运行的平均值
        avg_uniformity = np.mean(uniformity_list)
        avg_constraint_violation = np.mean(constraint_violation_list)
        avg_final_sample_count = np.mean(final_sample_count_list)

        # 计算标准差（可选，反映结果稳定性）
        std_uniformity = np.std(uniformity_list)
        std_constraint_violation = np.std(constraint_violation_list)

        # 打印最终统计结果
        print(f"\n{'=' * 50}")
        print(f"{problem['name']} 10次运行统计结果:")
        print(f"{'=' * 50}")
        print(f"平均最终样本数: {avg_final_sample_count:.2f} (±{np.std(final_sample_count_list):.2f})")
        print(f"平均均匀性: {avg_uniformity:.6f} (±{std_uniformity:.6f})")
        print(f"平均约束违反程度: {avg_constraint_violation:.6f} (±{std_constraint_violation:.6f})")
        print(f"{'=' * 50}\n")
