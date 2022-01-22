from .poly_interpolation import *

"""
polynomial operator over 1 float variable
"""
class SPoly:

    """
    v := vector of values, length l -1 is power, index 0 is greatest power
    """
    def __init__(self,v):
        assert is_vector(v), "invalid vector"
        self.v = v

    def apply(self,x):
        s = 0.0
        l = len(self.v) - 1
        for v_ in self.v:
            s += (v_ * x ** l)
            l -= 1
        return s

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
