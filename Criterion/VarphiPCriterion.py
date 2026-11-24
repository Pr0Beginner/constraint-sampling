import numpy as np
from scipy.spatial.distance import pdist

from Criterion.PhiPCriterion import PhiPCriterion


# 这是一个占位符基类，以便示例可以独立运行。
class Criterion:
    def evaluate(self, H):
        raise NotImplementedError


# 假设上面已定义了修正后的 PhiPCriterion 类
class PhiP1dCriterion(Criterion):
    """
    计算一维投影准则 (Phi_p_1d)。

    该准则在 Wu et al. (2019) 的论文中有定义，用于衡量样本点
    在每个坐标轴上投影的均匀性。[cite: 313, 548, 551]
    """

    def __init__(self, p_exponent=2):
        """
        初始化准则。

        Args:
            p_exponent (int): 用于底层 PhiP 计算的指数 'p'。默认为 2。
        """
        # 对于一维数据，距离范数总是绝对差，
        # 因此在底层的 PhiPCriterion 中 p_norm=1。
        self.phi_p_calculator = PhiPCriterion(p_exponent=p_exponent, p_norm=1)

    def evaluate(self, H: np.ndarray) -> float:
        """
        计算 Phi_p_1d 准则。

        Args:
            H (np.ndarray): 形状为 (点数, 维度数) 的设计矩阵。

        Returns:
            float: Phi_p_1d 的值。值越小越好。
        """
        if not isinstance(H, np.ndarray) or H.ndim != 2:
            raise ValueError("输入 H 必须是一个二维 numpy 数组。")

        n_points, n_dims = H.shape

        if n_points < 2:
            return 0.0

        phi_p_per_dim = np.zeros(n_dims)

        for k in range(n_dims):
            # 提取第 k 列并将其重塑为二维数组 (n, 1)，
            # 以便我们的 PhiPCriterion 类可以处理它。
            column = H[:, k].reshape(-1, 1)
            phi_p_per_dim[k] = self.phi_p_calculator.evaluate(column)

        return np.max(phi_p_per_dim)
