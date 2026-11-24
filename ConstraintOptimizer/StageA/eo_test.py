import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import casadi
import pandas as pd

# 将项目根目录加入 sys.path，解决直接执行脚本时的模块导入问题
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from matplotlib import font_manager
from ConstraintOptimizer.StageA.EvolutionaryOptimizer import EvolutionaryOptimizer
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

# E3
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


def get_cec_g25_problem():
    # 变量定义
    factors = {
        'x1': {'type': 'cont', 'bounds': [0, 230]},  # 裂纹中点 x
        'x2': {'type': 'cont', 'bounds': [0, 50]},  # 裂纹中点 y
        'x3': {'type': 'cont', 'bounds': [10, 30]},  # 裂纹半径
        'x4': {'type': 'cont', 'bounds': [0, np.pi]},  # 裂纹角度
    }
    # 辅助函数：获取裂纹的两个端点 P1, P2
    def get_endpoints(x):
        r = x['x3']
        theta = x['x4']
        dx = r * casadi.cos(theta)
        dy = r * casadi.sin(theta)

        # P1 = Midpoint + Delta
        p1_x = x['x1'] + dx
        p1_y = x['x2'] + dy

        # P2 = Midpoint - Delta
        p2_x = x['x1'] - dx
        p2_y = x['x2'] - dy
        return (p1_x, p1_y), (p2_x, p2_y)
    # 边界直线定义 Ax + By + C = 0
    lines = [
        (1, 0, 0),  # L1
        (1, 0, -230),  # L2
        (0, 1, 0),  # L3
        (0, 1, -50),  # L4
        (-0.043, 1, 1.29),  # L5
        (0.11151, 1, -53.345)  # L6
    ]

    constraints = []

    # --- 几何包含约束 (Table 6, g1 - g11) ---

    # g1: 中点 x['x1'] >= 0 (L1 右侧)
    constraints.append({'type': 'ineq', 'fun': lambda x: x['x1']})

    # g2, g3: P1, P2 在 L2 (x=230) 左侧 -> 230 - x >= 0
    constraints.append({'type': 'ineq', 'fun': lambda x: 230 - get_endpoints(x)[0][0]})
    constraints.append({'type': 'ineq', 'fun': lambda x: 230 - get_endpoints(x)[1][0]})

    # g4, g5: P1, P2 在 L3 (y=0) 上方 -> y >= 0
    constraints.append({'type': 'ineq', 'fun': lambda x: get_endpoints(x)[0][1]})
    constraints.append({'type': 'ineq', 'fun': lambda x: get_endpoints(x)[1][1]})

    # g6, g7: P1, P2 在 L4 (y=50) 下方 -> 50 - y >= 0
    constraints.append({'type': 'ineq', 'fun': lambda x: 50 - get_endpoints(x)[0][1]})
    constraints.append({'type': 'ineq', 'fun': lambda x: 50 - get_endpoints(x)[1][1]})

    # g8, g9: P1, P2 在 L5 (y - 0.043x + 1.29 = 0) 上方
    def con_L5(p):
        return p[1] - 0.043 * p[0] + 1.29

    constraints.append({'type': 'ineq', 'fun': lambda x: con_L5(get_endpoints(x)[0])})
    constraints.append({'type': 'ineq', 'fun': lambda x: con_L5(get_endpoints(x)[1])})

    # g10, g11: P1, P2 在 L6 (y + 0.1115x - 53.345 = 0) 下方
    def con_L6(p):
        return 53.345 - 0.11151 * p[0] - p[1]

    constraints.append({'type': 'ineq', 'fun': lambda x: con_L6(get_endpoints(x)[0])})
    constraints.append({'type': 'ineq', 'fun': lambda x: con_L6(get_endpoints(x)[1])})

    # --- 结构安全/距离约束 (Table 6, g12) ---
    # 公式 (31): min_{i!=j} (dist(P1, Li) + dist(P2, Lj)) >= 4

    def dist_constraint(x):
        (p1x, p1y), (p2x, p2y) = get_endpoints(x)

        # 计算 P1 到所有 6 条线的距离
        dists1 = []
        dists2 = []

        for (A, B, C) in lines:
            denom = np.sqrt(A ** 2 + B ** 2)
            # 修正：使用 casadi.fabs 而不是 casadi.abs
            dists1.append(casadi.fabs(A * p1x + B * p1y + C) / denom)
            dists2.append(casadi.fabs(A * p2x + B * p2y + C) / denom)

        # 组合所有 i != j 的距离和
        pair_sums = []
        for i in range(len(lines)):
            for j in range(len(lines)):
                # 排除 L1 (中轴)，且要求是不同边界
                if i == 0 or j == 0:
                    continue
                if i != j:
                    pair_sums.append(dists1[i] + dists2[j])

        # 使用 casadi.vertcat 将列表转换为向量，然后用 mmin 求最小值
        if not pair_sums:  # 防止空列表（理论上不会发生）
            return 0

        all_sums_vec = casadi.vertcat(*pair_sums)
        min_dist_sum = casadi.mmin(all_sums_vec)

        return min_dist_sum - 4.0

    constraints.append({'type': 'ineq', 'fun': dist_constraint})

    return {"name": "g25_aircraft_crack", "latent_dim": 4, "factors": factors, "constraints": constraints}


def E1_problem():
    """
    实验 E1：2 变量，边界 [0,1]，约束 1-(x1-1)^2-(x2-1)^2 <= 0
    格式完全参照 get_cec_g24_problem
    """
    factors = {
        'x1': {'type': 'cont', 'bounds': [0, 1]},
        'x2': {'type': 'cont', 'bounds': [0, 1]}
    }

    constraints = [
        {
            'type': 'ineq',
            'fun': lambda x: -(1 - (x['x1'] - 1) ** 2 - (x['x2'] - 1) ** 2)
        }
    ]

    return {
        "name": "E1",
        "latent_dim": 2,
        "factors": factors,
        "constraints": constraints
    }

import casadi as ca
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
import matplotlib.patches as mpatches


def plot_E1_feasible_region(final_samples=None, save_path='E1_feasible_region_with_samples.png'):
    fig, ax = plt.subplots(figsize=(7, 7))
    x1_min, x1_max = 0, 1
    x2_min, x2_max = 0, 1

    x1 = np.linspace(x1_min, x1_max, 800)
    x2 = np.linspace(x2_min, x2_max, 800)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1 - 1) ** 2 + (X2 - 1) ** 2 - 1

    ax.contourf(X1, X2, Z, levels=[0, np.inf], colors=['#2ca02c'], alpha=0.6)
    ax.contourf(X1, X2, Z, levels=[-np.inf, 0], colors=['#e0e0e0'], alpha=0.4)
    ax.contour(X1, X2, Z, levels=[0], colors=['red'], linewidths=2.5)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(rect)

    handles = [
        mpatches.Patch(color='#2ca02c', alpha=0.6, label='Feasible Region'),
        mpatches.Patch(color='#e0e0e0', alpha=0.4, label='Infeasible Region'),
        plt.Line2D([0], [0], color='red', lw=2.5, label='Constraint Boundary: $g(x)=0$'),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Sample Space $[0,1]^2$')
    ]

    if final_samples is not None:
        final_samples = np.array(final_samples)
        scatter = ax.scatter(final_samples[:, 0], final_samples[:, 1],
                             color='black', s=40, zorder=5, label='Samples', marker='o')
        handles.append(scatter)

    ax.legend(handles=handles, loc='upper right')
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)

    # ✅ 修复标题
    if final_samples is not None:
        ax.set_title(f'E1 with {len(final_samples)} samples', fontsize=16, pad=15)
    else:
        ax.set_title('E1 feasible region', fontsize=16, pad=15)

    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ E1 图已保存: {save_path}")

def plot_E2_feasible_region(final_samples=None, save_path='E2_feasible_region_with_samples.png'):
    """
    绘制 E2 问题的样本空间、约束边界、可行域，并将优化后的样本点绘制在图上。

    参数:
        final_samples (np.ndarray or list): 优化器返回的最终样本点，形状应为 (N, 2)。
        save_path (str): 生成图片的保存路径
    """
    # 创建图形并设置大小
    fig, ax = plt.subplots(figsize=(7, 7))

    # 样本空间边界 [0, 1] x [0, 1]
    x1_min, x1_max = 0, 1
    x2_min, x2_max = 0, 1

    # 生成高精度网格数据
    x1 = np.linspace(x1_min, x1_max, 800)
    x2 = np.linspace(x2_min, x2_max, 800)
    X1, X2 = np.meshgrid(x1, x2)

    # 计算约束函数值 g(x) = min{max{0.5-x1, 0.5-x2}, x1^2+x2^2-0.25}
    part1 = np.maximum(0.5 - X1, 0.5 - X2)
    part2 = X1**2 + X2**2 - 0.25
    g = np.minimum(part1, part2)

    # 可行域条件：g(x) <= 0
    # 为了contourf的层级设置，我们定义Z = -g(x)，则可行域对应Z >= 0
    Z = -g

    # 绘制可行域（绿色区域，Z >= 0）
    ax.contourf(X1, X2, Z, levels=[0, np.inf], colors=['#2ca02c'], alpha=0.6)

    # 绘制不可行域（灰色区域，Z < 0）
    ax.contourf(X1, X2, Z, levels=[-np.inf, 0], colors=['#e0e0e0'], alpha=0.4)

    # 绘制约束边界（红色线条，Z == 0）
    ax.contour(X1, X2, Z, levels=[0], colors=['red'], linewidths=2.5)

    # 设置略大于样本空间的坐标轴，以避免边界线贴在图像最边缘
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # 绘制样本空间的黑色虚线框
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(rect)

    # 定制基础图例
    handles = [
        mpatches.Patch(color='#2ca02c', alpha=0.6, label='Feasible Region'),
        mpatches.Patch(color='#e0e0e0', alpha=0.4, label='Infeasible Region'),
        plt.Line2D([0], [0], color='red', lw=2.5, label='Constraint Boundary: $g(x)=0$'),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Sample Space $[0,1]^2$')
    ]

    # 绘制样本点（如果传入）
    if final_samples is not None:
        final_samples = np.array(final_samples)
        # zorder=5 确保散点被画在多边形和网格线的上方
        scatter = ax.scatter(final_samples[:, 0], final_samples[:, 1],
                             color='black', s=40, zorder=5, label='Samples', marker='o')
        handles.append(scatter)  # 将样本点的图例加入列表

    # 添加图例
    ax.legend(handles=handles, loc='upper left')

    # 设置标签与标题
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    if final_samples is not None:
        ax.set_title(f'E2 with {len(final_samples)} samples', fontsize=16, pad=15)
    else:
        ax.set_title('E2 Feasible Region', fontsize=16, pad=15)

    # 添加网格与调整比例
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')

    # 保存并关闭图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 带样本点的 E2 问题可视化图片已成功保存至: {save_path}")

if __name__ == "__main__":
    # 包含所有能求解的约束问题
    # inequality_problems = [
    #     get_cec_g01_problem(), get_cec_g02_problem(), get_cec_g04_problem(),
    #     get_cec_g06_problem(), get_cec_g07_problem(), get_cec_g08_problem(),
    #     get_cec_g09_problem(), get_cec_g10_problem(), get_cec_g16_problem(),
    #     get_cec_g18_problem(), get_cec_g19_problem(), get_cec_g24_problem(),
    # ]
    inequality_problems = [
        get_cec_g01_problem(), get_cec_g04_problem(), get_cec_g09_problem(),E1_problem(),E2_problem()
    ]

    # sample_nums = [20, 50, 100]
    sample_nums = [20, 50]

    # 最外层循环：先遍历所有样本数设定
    for sample_num in sample_nums:
        print(f"\n{'#' * 60}")
        print(f"🚀 开始全面测试，当前设定的样本数 (sample_num) 为：{sample_num}")
        print(f"{'#' * 60}\n")

        # 内层循环：遍历所有问题
        for problem in inequality_problems:
            print(f"{'=' * 50}")
            print(f"正在测试: {problem['name']} (样本数: {sample_num}, 共运行 10 次，不打印单次日志)...")

            # 数据收集容器（必须在这里清空，确保不同样本数、不同问题的数据独立）
            uniformity_list = []
            constraint_violation_list = []
            final_sample_count_list = []
            elapsed_time_list = []

            # 跟踪当次问题和样本数设定下，表现最好的一次以导出 Excel
            best_score = float('inf')
            best_history = []

            # 循环测试 2 次
            for run_idx in range(2):
                optimizer = EvolutionaryOptimizer(
                    factors=problem['factors'],
                    constraints=problem['constraints'],
                    max_generations=1500,
                    sample_num=sample_num,
                    isDraw=False  # 禁用可视化提升速度
                )

                t_start = time.time()
                final_samples = optimizer.optimize()
                elapsed = time.time() - t_start

                if problem["name"] == "E1":
                    print('start draw E1')
                    plot_E1_feasible_region(final_samples, f'E1_{sample_num}.png')
                if problem["name"] == "E2":
                    print('start draw E2')
                    plot_E2_feasible_region(final_samples, f'E2_{sample_num}.png')
                # 计算当次核心指标
                final_sample_count = len(final_samples)
                uniformity = optimizer.uniformity_evaluator.evaluate(final_samples)
                constraint_violation = np.sum([
                    optimizer.constraint_handler.cal_val(sample)
                    for sample in optimizer.elite_population
                ])

                # 存入列表
                final_sample_count_list.append(final_sample_count)
                uniformity_list.append(uniformity)
                constraint_violation_list.append(constraint_violation)
                elapsed_time_list.append(elapsed)

                # 记录表现最好的一次（约束违规最小优先，其次是准则值最小）
                score = constraint_violation * 1e6 + uniformity
                if score < best_score:
                    best_score = score
                    best_history = optimizer.history.copy()

            # 10次循环结束，计算并输出当前 (问题, 样本数) 组合的总统计结果
            avg_uniformity = np.mean(uniformity_list)
            avg_constraint_violation = np.mean(constraint_violation_list)
            avg_final_sample_count = np.mean(final_sample_count_list)

            std_uniformity = np.std(uniformity_list)
            std_constraint_violation = np.std(constraint_violation_list)

            avg_elapsed_time = np.mean(elapsed_time_list)

            print(f"\n✅ {problem['name']} (样本数：{sample_num}) 2次运行统计结果:")
            print(f"平均最终样本数: {avg_final_sample_count:.2f} (±{np.std(final_sample_count_list):.2f})")
            print(f"平均均匀性: {avg_uniformity:.6f} (±{std_uniformity:.6f})")
            print(f"平均约束违反程度: {avg_constraint_violation:.6f} (±{std_constraint_violation:.6f})")
            print(f"平均用时: {avg_elapsed_time:.2f}s (±{np.std(elapsed_time_list):.2f}s)")

            # 将最优的一次收敛历史导出为 Excel
            if best_history:
                df = pd.DataFrame({
                    'Iteration': range(1, len(best_history) + 1),
                    'Criterion_Value': best_history
                })
                excel_filename = f"{problem['name']}_{sample_num}_best_history.xlsx"
                df.to_excel(excel_filename, index=False)
                print(f"📁 已成功导出最优收敛历史数据至: {excel_filename}")

            print(f"{'=' * 50}\n")
