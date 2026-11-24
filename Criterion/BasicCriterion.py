from abc import ABC, abstractmethod
import sys
import numpy as np
import math

class Criterion(ABC):
    """
    抽象基类(Abstract Base Class)，定义抽样准则的接口
    ABC是Python中用于创建抽象基类的模块，使用@abstractmethod装饰器
    可以强制子类必须实现某些方法，确保接口的一致性。
    """

    def calculate_distance_sum(self, population, i):
        """计算样本i到其他样本的最小距离"""
        reference_point = population[i]
        min_distance = sys.float_info.max
        for idx in range(len(population)):
            sample = population[idx]
            if idx != i:
                # 判断样本是否与参考点完全相同
                if np.all(sample == reference_point):
                    distance = sys.float_info.max
                else:
                    # 计算两点间的欧几里得距离
                    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(reference_point, sample)))
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def calculate_min_distance(self, population):
        """计算种群中所有样本的最小距离和"""
        min_dist = sys.float_info.max
        for i in range(len(population)):
            distance_sum = 0
            reference_point = population[i]
            for sample in population:
                if not np.all(sample == reference_point):
                    # 计算两点间的欧几里得距离
                    distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(reference_point, sample)))
                    distance_sum += distance
            if min_dist > distance_sum:
                min_dist = distance_sum
        return min_dist

    @staticmethod
    def quick_pow(x: np.ndarray, p: int):
        """快速幂运算"""
        result = np.ones_like(x)
        base = x.copy()
        exp = p

        while exp > 0:
            if exp & 1:
                result *= base
            base *= base
            exp >>= 1
        return result

    @abstractmethod
    def evaluate(self, design):
        """评估设计矩阵的质量(子类必须实现)"""
        pass