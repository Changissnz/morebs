from .numerical_generator import * 
from .measures import * 
from types import MethodType,FunctionType
from copy import deepcopy

VECTOR_PIECEWISE_ADDITIVE_INFO_MODES = {"sign-1","sign-all","value-1","value-all"} 
DEFAULT_MIDDLE_SEGMENT_DELTA_RATIO = [0.05,0.9] 

# TODO: test this. 
"""
Calculates `segment_size` vectors such that sum (net) of those vectors equals `sum_info`, 
either a float singleton or a vector of `length` elements. 

Uses a PRNG to calculate those additive vectors. 
"""
class VectorPiecewiseAdditiveDerivative: 

    def __init__(self,length,sum_info,prg,segment_size:int=5,record_derivative_info=None):
        assert type(length) == int 
        assert length > 0 
        assert type(prg) in {MethodType,FunctionType}

        self.length = length 
        self.sum_info = None  
        self.is_sum_single = None
        self.seg_size = None 

        self.record = [] 
        self.record_di = None 

        self.prg = prg 
        self.set_sum_info(sum_info) 
        self.set_segment_size(segment_size) 
        self.set_record(record_derivative_info)

        self.c = 0 
        self.cumulative_delta = np.zeros((self.length,),dtype=float)
        self.prev = None 
        self.now = deepcopy(self.cumulative_delta) 
        return 

    def set_sum_info(self,sum_info): 
        single_stat = is_number(sum_info)
        if not single_stat: 
            assert is_vector(sum_info)
        self.sum_info = sum_info 
        self.is_sum_single = single_stat 

    def set_segment_size(self,seg_size:int): 
        assert type(seg_size) == int 
        assert seg_size > 0 
        self.seg_size = seg_size 
        return 

    def set_record(self,record_derivative_info:bool): 
        assert type(record_derivative_info) == bool 
        self.record_di = record_derivative_info

    def reset(self,sum_info,seg_size:int): 
        self.set_sum_info(sum_info)
        self.set_segment_size(seg_size) 
        self.c = 0 
        self.cumulative_delta = np.zeros((self.length,),dtype=float)
        self.record = [] 

    def __next__(self): 
        if self.c >= self.seg_size: 
            return 

        # case: last segment 
        is_last = self.c + 1 == self.seg_size

        if self.is_sum_single: 
            q = self.process_single(is_last)
        else: 
            q = self.process_vec(is_last) 

        self.cumulative_delta += q 
        self.c += 1 

        if self.record_di:
            self.record.append(q) 

        self.prev = self.now
        self.now = deepcopy(q) 
        return q 

    def process_single(self,is_last:bool): 

        x = self.cumulative_diff() 

        if not is_last: 
            r = prg_decimal(self.prg,DEFAULT_MIDDLE_SEGMENT_DELTA_RATIO) 
            x = round(x * r ,5) 

        prg_ = prg__single_to_int(self.prg)
        variance = prg_decimal(self.prg,[0.,1.])
        n = modulo_in_range(int(self.prg()),[500,1500]) 
        P = prg_partition_for_float(x,self.length,prg_,variance,n=n,rounding_depth=10)
        return P 

    def process_vec(self,is_last:bool): 

        d = self.cumulative_diff() 
        if is_last: 
            return d

        v = np.zeros((self.length,),dtype=float) 

        for i in range(self.length): 
            r = prg_decimal(self.prg,DEFAULT_MIDDLE_SEGMENT_DELTA_RATIO)
            sign = prg_decimal(self.prg,DEFAULT_MIDDLE_SEGMENT_DELTA_RATIO)

            if sign >= 0.5: 
                v[i] = r
            else: 
                v[i] = -r 

        return np.round(d * v,5) 
         
    def cumulative_diff(self): 
        # case: single 
        if self.is_sum_single: 
            s = np.sum(self.cumulative_delta) 
            return round(self.sum_info - s,5) 
        return np.round(self.sum_info - self.cumulative_delta,5) 

    def derivative_info(self,info_type):
        assert info_type in VECTOR_PIECEWISE_ADDITIVE_INFO_MODES
        assert self.c > 0 

        if info_type in {"sign-1","sign-all"}: 
            q = to_trinary_relation_v2(self.now,self.prev)

            if info_type == "sign-all": 
                return q 
            
            c = Counter(q)
            c = [(k,v) for k,v in c.items()] 
            c = sorted(c,key=lambda x:x[1],reverse=True) 
            if len(c) == 1: 
                return c[0][0] 
            
            # case: tie, return 0 
            if c[0][1] == c[1][1]: 
                return 0 
            return c[0][0] 

        s = self.now - self.prev 

        if info_type == "value-1": 
            return np.sum(s) 
        return s  