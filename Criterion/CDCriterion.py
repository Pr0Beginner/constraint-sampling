from Criterion.BasicCriterion import Criterion
import math
'''
    代码来自：https://github.com/gongw03/MO-ASMO
'''

class CDCriterion(Criterion):
    """基于PhiP准则的评估"""

    def __init__(self):
        pass
    def evaluate(self, X):
        ''' Centered L2-discrepancy'''
        num, dim = X.shape
        D1 = (13.0/12.0)**dim

        D2 = 0.0
        for k in range(num):
            DD2 = 1.0
            for j in range(dim):
                DD2 = DD2 * (1 + 0.5*abs(X[k,j]-0.5) - 0.5*abs(X[k,j]-0.5)**2)
            D2 = D2 + DD2

        D3 = 0.0
        for k in range(num):
            for j in range(num):
                DD3 = 1.0
                for i in range(dim):
                    DD3 = DD3 * (1 + 0.5*abs(X[k,i]-0.5) + 0.5*abs(X[j,i]-0.5) - 0.5*abs(X[k,i]-X[j,i]))
                D3 = D3 + DD3

        D = math.sqrt(D1 + D2*(-2.0/num) + D3/(num**2))
        return D

