import math
from Criterion.BasicCriterion import Criterion

'''
    代码来自：https://github.com/gongw03/MO-ASMO
'''

class WDCriterion(Criterion):
    """基于PhiP准则的评估"""

    def __init__(self):
        pass
    def evaluate(X):
        ''' Wrap-around L2-discrepancy'''
        num, dim = X.shape
        D1 = -(4.0/3.0)**dim

        D3 = 0.0
        for k in range(num):
            for j in range(num):
                DD3 = 1.0
                for i in range(dim):
                    DD3 = DD3 * (3.0/2.0 - abs(X[k,i]-X[j,i]) * (1 - abs(X[k,i]-X[j,i])))
                D3 = D3 + DD3
        D = math.sqrt(D1 + D3/(num**2))
        return D