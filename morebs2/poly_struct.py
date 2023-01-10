from .poly_interpolation import *
from copy import deepcopy
import numpy as np


"""
"""
class SPoly:
    """
    polynomial operator over 1 float variable


    :param v: vector of values, length l -1 is power, index 0 is greatest power
    """

    def __init__(self,v):
        assert is_vector(v), "invalid vector"
        self.v = v

    def __str__(self):
        y = self.ce_notation()
        s = ""
        for y_ in y:
            s += " +" if y_[0] >= 0 else " "
            s += " {}x^{}".format(y_[0],y_[1])
        if len(s) == 0:
            return s
        return s[3:]

    def apply(self,x):
        s = 0.0
        l = len(self.v) - 1
        for v_ in self.v:
            s += (v_ * x ** l)
            l -= 1
        return s

    def __mul__(self,s):
        l2 = len(self.v) -1 + len(s.v)
        v = np.zeros(l2)
        v2 = np.zeros(l2)

        le1 = len(self.v) - 1
        le2 = len(s.v) - 1

        for (i,x) in enumerate(self.v):
            v3 = deepcopy(v2)
            e1 = le1 - i
            for (j,x2) in enumerate(s.v):
                x3 = x * x2
                e2 = le2 - j
                e3 = e1 + e2
                v3[l2 - e3 - 1] = x3
            v = v + v3
        return SPoly(v) 

    def vector_index_notation(self):
        return deepcopy(self.v)

    def ce_notation(self):
        '''
        coefficient-exponent notation
        '''
        p = []
        l = len(self.v) - 1 

        for (i,v_) in enumerate(self.v):
            if v_ == 0.:
                continue
            p.append((v_,l - i))
        return np.array(p)

    @staticmethod
    def from_ce_notation(v):
        return "TO DO"

"""
inversion of SPoly
"""
class ISPoly:

    """
    v := vector of values, length l -1 is power, index 0 is greatest power
    """
    def __init__(self,x):
        assert type(x) == float, "invalid x"
        self.x = x

    def apply(self,v):
        q = SPoly(v)
        return q.apply(self.x)

def multiply_polynomials_ce_notation(p1,p2):
    '''
    '''
    p = []



    return -1