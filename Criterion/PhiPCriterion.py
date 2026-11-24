import numpy as np
from scipy.spatial.distance import pdist


# 这是一个占位符基类，以便示例可以独立运行。
class Criterion:
    def evaluate(self, H):
        raise NotImplementedError


class PhiPCriterion(Criterion):
    """
    计算给定设计矩阵的 PhiP 准则。

    该准则在 Wu et al. (2019) 和 You et al. (2021) 的论文中有定义。
    [cite: 313, 7]
    """

    def __init__(self, p_exponent=2, p_norm=2):
        """
        初始化准则。

        Args:
            p_exponent (int): Phi_p 公式中的指数 'p'。默认为 2。
            p_norm (int): 闵可夫斯基距离范数的阶数
                          (1 代表曼哈顿距离, 2 代表欧几里得距离)。默认为 2。
        """
        if p_exponent <= 0:
            raise ValueError("p_exponent 必须是一个正数。")
        self.p_exponent = p_exponent
        self.p_norm = p_norm

    def evaluate(self, H: np.ndarray) -> float:
        """
        计算 Phi_p 空间填充准则。

        Args:
            H (np.ndarray): 形状为 (点数, 维度数) 的设计矩阵。

        Returns:
            float: Phi_p 的值。值越小越好。
        """
        if not isinstance(H, np.ndarray) or H.ndim != 2:
            raise ValueError("输入 H 必须是一个二维 numpy 数组。")

        # pdist 计算 H 中各点之间的成对距离的压缩形式。
        # 'minkowski' 度量配合 'p=p_norm' 参数可以使用欧几里得 (p=2)、
        # 曼哈顿 (p=1) 等距离。
        try:
            distances = pdist(H, metric='minkowski', p=self.p_norm)
        except Exception:
            print(H)

        # 检查是否存在零距离，这会导致除以零的错误。
        if np.any(distances == 0):
            return np.inf

        phi_p_val = (np.sum(distances ** (-self.p_exponent))) ** (1.0 / self.p_exponent)

        return phi_p_val
