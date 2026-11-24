import numpy as np
from scipy.spatial.distance import pdist


# 这是一个占位符基类，以便示例可以独立运行。
class Criterion:
    def evaluate(self, H):
        raise NotImplementedError


class MaximinCriterion(Criterion):
    """
    计算 maximin 距离准则。

    该准则衡量设计中任意两点间的最小距离。一个好的 maximin 设计
    目标是最大化这个值。该准则在 Geng et al. (2020) 中被引用。
    [cite: 1000]
    """

    def __init__(self, p_norm=2):
        """
        初始化准则。

        Args:
            p_norm (int): 闵可夫斯基距离范数的阶数
                          (1 代表曼哈顿距离, 2 代表欧几里得距离)。默认为 2。
        """
        self.p_norm = p_norm

    def evaluate(self, H: np.ndarray) -> float:
        """
        计算最小点间距离。

        Args:
            H (np.ndarray): 形状为 (点数, 维度数) 的设计矩阵。

        Returns:
            float: 最小距离。值越大越好, 但是这里为了统一，加上负号。
        """
        if not isinstance(H, np.ndarray) or H.ndim != 2:
            raise ValueError("输入 H 必须是一个二维 numpy 数组。")

        if H.shape[0] < 2:
            # 如果只有一个点或没有点，最小距离的概念没有明确定义。
            # 根据上下文可以返回 inf 或 0。返回 0 是一个安全的选择。
            return np.max

        distances = pdist(H, metric='minkowski', p=self.p_norm)

        return - 1 / np.min(distances)
