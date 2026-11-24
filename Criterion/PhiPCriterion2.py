import warnings

import numpy as np
from Criterion.BasicCriterion import Criterion

class PhiPCriterion2(Criterion):

    def __init__(self, q = 2):
        self.q = q

    def _jd(sefl,H,p=1):
        #number of points in the sampling plan
        n = H.shape[0]
        #computes the distances between all pairs of points
        d = np.zeros((n*(n-1)//2))
        list = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        for k, l in enumerate(list):
            d[k] = np.linalg.norm((H[l[0], :] - H[l[1], :]), p)
        # remove multiple occurences
        distinct_d, J = np.unique(d, return_counts=True)

        return J, distinct_d

    def evaluate(self, H):
        """
        Calculates the space-filling criterion
        from an open toolbox:pykring http://www.pykriging.com/
        """
        p = 2
        J, d = self._jd(H, p)
        Phiq = (np.sum(J * (d ** (-self.q)))) ** (1.0 / self.q)
        return Phiq
