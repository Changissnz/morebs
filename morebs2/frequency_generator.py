from .distributions import * 
from .numerical_generator import * 
from types import MethodType,FunctionType

class PoissonBasedFreqencyOutputter:

    def __init__(self,prg): 
        assert type(prg) in {MethodType,FunctionType}        
        self.prg = prg 
        return

    def out(self,expected,possible_range):
        assert is_valid_range(possible_range,True,False) 
        assert possible_range[0] <= expected < possible_range[1] 

        indices = [] 
        pr = [] 
        for i in range(possible_range[0],possible_range[1]):
            p = poisson_distribution_pr(expected,i)
            pr.append(p)
            indices.append(i) 

        pr = np.array(pr)
        x = np.sum(pr) 
        pr = pr / x 

        d = prg_decimal(self.prg,[0.,1.])
        ref = 0.0 
        while len(pr) > 0:
            p = pr[0] 
            pr = pr[1:]
            i = indices.pop(0)
            if ref <= d <= ref + p: 
                return i 
            ref = ref + p