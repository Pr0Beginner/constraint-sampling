from Criterion.BasicCriterion import Criterion
import math
class MaxProCriterion(Criterion):
    """
        最大投影评估准则
        参考论文：Joseph V R, Gul E, Ba S. Designing computer experiments with multiple types of factors: The MaxPro approach[J]. Journal of Quality Technology, 2020, 52(4): 343-354.
    """

    def __init__(self, cont_num, disc_num, cat_num):
        self.cont_num = cont_num
        self.disc_num = disc_num
        self.cat_num = cat_num

    def evaluate(self, Matrix):
        n = Matrix.shape[0]  # 样本数量
        p1 = self.cont_num
        p2 = self.disc_num
        p3 = self.cat_num
        if n != p1 + p2 + p3:
            raise ValueError("对象中因子数和矩阵中因子数不符合!")

        # 计算组合数 C(n,2)
        comb_n_2 = n * (n - 1) // 2

        # 初始化 d_ij, u_ij, v_ij 相关变量
        d_ij_list = []
        u_ij_list = []
        v_ij_list = []

        for i in range(n):
            for j in range(i + 1, n):
                # 计算 d_ij
                d_ij = 1.0
                for l in range(p1):
                    d_il = Matrix[i, l]
                    d_jl = Matrix[j, l]
                    d_ij *= (d_il - d_jl) ** 2
                d_ij = 1.0 / d_ij
                d_ij_list.append(d_ij)

                # 计算 u_ij
                u_ij = 1.0
                m_k = 1.0  # 这里假设 m_k 为 1，可根据实际情况修改
                for k in range(p2):
                    u_ik = Matrix[i, p1 + k]
                    u_jk = Matrix[j, p1 + k]
                    u_ij *= (abs(u_ik - u_jk) + 1.0 / m_k) ** 2
                u_ij = 1.0 / u_ij
                u_ij_list.append(u_ij)

                # 计算 v_ij
                v_ij = 1.0
                L_h = 1.0  # 这里假设 L_h 为 1，可根据实际情况修改
                for h in range(p3):
                    v_ih = Matrix[i, p1 + p2 + h]
                    v_jh = Matrix[j, p1 + p2 + h]
                    indicator = 1 if v_ih != v_jh else 0
                    v_ij *= (indicator + 1.0 / L_h) ** 2
                v_ij = 1.0 / v_ij
                v_ij_list.append(v_ij)

        # 计算中间求和项
        sum_term = 0
        for idx in range(comb_n_2):
            sum_term += d_ij_list[idx] * u_ij_list[idx] * v_ij_list[idx]

        # 计算组合数的倒数
        comb_reciprocal = 1.0 / comb_n_2

        # 计算 psi(D)
        exponent = 1.0 / (p1 + p2 + p3)
        psi_d = (comb_reciprocal * sum_term) ** exponent

        return psi_d

