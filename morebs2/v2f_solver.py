import numpy as np 
from copy import deepcopy 
from .matrix_methods import is_vector,is_valid_range,is_bounds_vector
from .numerical_generator import prg__single_to_int,prg__LCG,prg_choose_n,prg_decimal,\
    modulo_in_range,prg_seqsort

from types import FunctionType,MethodType 

DEFAULT_FIT_TYPE_1_COEFFICIENT_SCALE = [-2/1.2,2/1.2] 
DEFAULT_FIT_TYPE_1_MAX_ADJUSTMENT_STEPS = [1,1000] 

class Vector2FloatSolver: 

    def __init__(self,V,f,prg): 
        assert is_vector(V)
        assert type(prg) in {FunctionType,MethodType}
        
        self.V = V 
        self.f = float(f)
        self.prg = prg 
        self.W = np.zeros((len(self.V),)) 
        self.fin_stat = False  

    """
    main method 
    """
    def solve(self): 
        return 

    def __next__(self): 
        return 

    def output(self): 
        return sum(self.V * self.W) 

"""
Type (S)tochastic One.  

Given a vector V, calculate a vector V2 of length |V| s.t. 
    V * V2 = f.

Uses a PRNG `prg`, along with `coeff_scale`, a range that specifies the 
stochastic multiples m' applied to the running difference d between a solution 
(in progress) and wanted float `f`. For example, at index i in running solution S, 
running difference is d_i. PRNG chooses a multiple m_ in range `coeff_scale`. Then 
algorithm calculates a weight w_i such that for the new running solution S' (w_i included), 
the running difference d_i' is 
    d_i' = d_i - (d_i * m_). 
"""
class Vector2FloatSolverTypeS1(Vector2FloatSolver):  

    def __init__(self,V,f,prg,coeff_scale=DEFAULT_FIT_TYPE_1_COEFFICIENT_SCALE,adjustment_steps=None):
        super().__init__(V,f,prg)
        assert is_valid_range(coeff_scale,False,True)

        self.coeff_scale = coeff_scale
        
        self.adjustment_indices = None 
        self.running_sum = 0
        self.adjustment_steps = adjustment_steps

        self.c = 0
        self.update_log = [] 
        self.initial_weights() 

    """
    main method 
    """
    def solve(self): 
        while not self.fin_stat: 
            next(self) 


    def __next__(self): 
        if self.fin_stat: 
            return None 

        self.c += 1 
        fit_exact = False 
        if self.c == self.adjustment_steps: 
             fit_exact = True 
             self.fin_stat = True 
        self.fit_next(fit_exact) 

    def output(self): 
        return sum(self.V * self.W) 


    def initial_weights(self): 
        self.W = self.W + 1 
        S = [i for i in range(len(self.V))]
        self.adjustment_indices = self.choose_non_zero_sum_subset(S) 
        self.number_of_adjustment_steps() 
        self.running_sum = sum(self.V * self.W) 

    def subset_coefficient_sum(self,ai): 
        return sum([self.V[i] for i in ai]) 

    def choose_non_zero_sum_subset(self,S,num_attempts=50): 
        if num_attempts < 0: assert False 

        prg_ = prg__single_to_int(self.prg)
        n = modulo_in_range(prg_(),[1,len(S)+1]) 
        q = prg_choose_n(deepcopy(S),n,\
            prg_,is_unique_picker=True)
        stat = round(self.subset_coefficient_sum(q),5) == 0 
        if stat: 
            return self.choose_non_zero_sum_subset(S,num_attempts-1)
        return q 

    def number_of_adjustment_steps(self):
        if type(self.adjustment_steps) == int and self.adjustment_steps > 0: 
            return 
        prg_ = prg__single_to_int(self.prg) 
        self.adjustment_steps = modulo_in_range(prg_(),DEFAULT_FIT_TYPE_1_MAX_ADJUSTMENT_STEPS)
        return

    def fit_next(self,fit_exact:bool): 
        s = 1.0 
        if not fit_exact: 
            s = modulo_in_range(self.prg(),self.coeff_scale)

        diff = self.f - self.output()
        diff = diff * s 

        stat = False 
        while not stat: 
            stat = self.choose_non_zero_sum_subset(self.adjustment_indices)

        q = self.choose_non_zero_sum_subset(self.adjustment_indices)
        cs = self.subset_coefficient_sum(q)

        q2 = diff / cs 

        self.update_vec(q2,q) 
        self.update_log.append((q2,q)) 
        return q2, q 

    def update_vec(self,f,indices): 
        for i in indices: 
            self.W[i] += f  
        return

"""
Type (R)ange(X). 

Given a vector V, calculate a vector V2 of length |V| s.t. 
    V * V2 = f.

For vector V2, every i'th element v_i falls in range 
`index_ranges[i]`. 
"""
class Vector2FloatSolverTypeRX(Vector2FloatSolver): 

    def __init__(self,V,index_ranges,f,prg):
        super().__init__(V,f,prg) 

        assert is_bounds_vector(index_ranges) 
        assert len(V) == index_ranges.shape[0] 

        self.index_ranges = index_ranges 

        self.scan_one = False  
        self.index = 0 

        self.running_diff = f 
        self.check_parameters() 

        self.adj_indices = None 
        self.adjustment_index_order() 

        self.attempt_exact = False 

    def check_parameters(self): 
        min_f = np.dot(self.V,self.index_ranges[:,0]) 
        max_f = np.dot(self.V,self.index_ranges[:,1]) 
        assert min_f <= self.f < max_f  

    def adjustment_index_order(self): 
        self.adj_indices = prg_seqsort([i for i in range(len(self.V))],self.prg)

    def solve(self,prelim_steps = 10,post_steps=10): 

        while not self.scan_one: 
            next(self) 

        while prelim_steps > 0: 
            self.next__scan_one() 
            prelim_steps -= 1 

        self.attempt_exact = True 

        while not self.fin_stat and post_steps > 0: 
            next(self) 
            post_steps -= 1 

        return

    def __next__(self): 

        if not self.scan_one: 
            self.next__scan_one() 
            return 

        self.next__adjust(self.attempt_exact)  
        return

    def next__scan_one(self): 
        if self.scan_one: return 

        r = self.index_ranges[self.index] 
        r_ = round(modulo_in_range(self.prg(),r),5) 
        self.W[self.index] = r_ 
        q = self.V[self.index] * r_ 
        self.running_diff -= round(q,5)  

        self.index += 1 
        if self.index >= len(self.V): 
            self.scan_one = True 
            self.index = 0 
        return

    def next__adjust(self,exact:bool):  
        if self.running_diff == 0.: 
            self.fin_stat = True 
            return True 

        r = self.index_ranges[self.index]

        index = self.index 
        self.index = (self.index + 1) % len(self.V)

        # case: 0 weight 
        if self.V[index] == 0: 
            print("NO @ ",index) 
            return False 

        # over or under 
            # case: decrease weight 
        if self.running_diff > 0: 
            # case: can't be decreased anymore 
            if self.W[index] == r[0]: 
                return False 
        else: 
            # case: can't be increased anymore 
            if self.W[index] == r[1]: 
                return False 

        q = self.generate_next_weight(index,exact) 
        x = self.W[index] * self.V[index] 
        x2 = q * self.V[index] 

        self.W[index] = q  
        self.running_diff = round(self.running_diff + x - x2,5)

        return False 

    def generate_next_weight(self,index,exact:bool):         
        
        r = self.index_ranges[index] 
        if not exact: 
            q = round(modulo_in_range(self.prg(),[r[0],self.W[index]]),5) 
        else: 

            w = self.running_diff / self.V[index] 
            q = round(self.W[index] + w,5) 

            if q < r[0]: 
                q = r[0] 
            elif q > r[1]: 
                q = r[1] 

        return q 
