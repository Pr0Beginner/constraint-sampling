import numpy as np
from scipy.stats import qmc
import time

# 适配原有的模块结构
if __package__:
    from .FactorEncoder import FactorEncoder
    from .ConstraintHandler import ConstraintHandler
else:
    from FactorEncoder import FactorEncoder
    from ConstraintHandler import ConstraintHandler

from Criterion.PhiPCriterion import PhiPCriterion


class LHDRejectionSampler:
    def __init__(self, factors, constraints, sample_num, max_evaluations=1000000):
        """
        基于拉丁超立方的拒绝采样器 (Baseline)

        :param factors: 变量定义字典
        :param constraints: 约束定义列表
        :param sample_num: 需要获取的目标可行解数量
        :param max_evaluations: 最大评估次数，防止可行域过小导致死循环
        """
        self.sample_num = sample_num
        self.max_evaluations = max_evaluations

        self.encoder = FactorEncoder(factors)
        # 容差保持与进化算法一致
        self.constraint_handler = ConstraintHandler(self.encoder, constraints, tolerance=1e-6)
        # 评价准则保持一致
        self.uniformity_evaluator = PhiPCriterion(p_exponent=2, p_norm=2)

        self.elite_population = []
        self.best_val = float('inf')
        self.history = []

    def optimize(self):
        """
        执行拒绝采样流程
        :return: 解码后的可行样本集 (numpy array)
        """
        sampler = qmc.LatinHypercube(d=self.encoder.total_dims)
        feasible_samples = []
        total_evaluated = 0

        # 批处理生成样本，提升效率
        batch_size = max(500, self.sample_num * 20)

        print(f"    -> 开始 LHD 拒绝采样，目标数量: {self.sample_num}...")
        start_time = time.time()

        while len(feasible_samples) < self.sample_num and total_evaluated < self.max_evaluations:
            # 1. 使用LHD生成批量 [0, 1] 空间的样本
            raw_samples = sampler.random(n=batch_size)
            samples = self.encoder.clip_and_round(raw_samples)

            # 2. 逐一验证可行性
            for p in samples:
                total_evaluated += 1

                # 若满足约束，则加入可行解集合
                if self.constraint_handler.judge_feasible(p):
                    feasible_samples.append(p)

                    if len(feasible_samples) >= self.sample_num:
                        break

                if total_evaluated >= self.max_evaluations:
                    break

        end_time = time.time()

        if len(feasible_samples) < self.sample_num:
            print(
                f"    [警告] 达到最大评估次数 ({self.max_evaluations})，仅找到 {len(feasible_samples)} 个可行解。可行域可能极小。")
        else:
            print(f"    -> 采样完成！耗时: {end_time - start_time:.2f}s，总评估次数: {total_evaluated}")

        # 截取所需数量的样本（防溢出）
        self.elite_population = feasible_samples[:self.sample_num]

        # 将样本解码回真实物理空间
        if len(self.elite_population) > 0:
            decoded_elites = self.encoder.decode_to_number(np.array(self.elite_population))
            # 计算最终的均匀性准则值
            self.best_val = self.uniformity_evaluator.evaluate(decoded_elites)
            self.history.append(self.best_val)  # 保持与 EO 接口兼容
        else:
            decoded_elites = np.array([])
            self.best_val = float('inf')

        return decoded_elites
