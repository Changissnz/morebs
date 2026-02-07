from .numerical_generator import * 
from types import MethodType,FunctionType
from copy import deepcopy

class XClassifier: 

    def __init__(self,D,L,index): 
        assert is_2dmatrix(D) 
        assert is_vector(L) 
        assert 0 <= index < D.shape[1] 

        self.D = D 
        self.D_ = deepcopy(D) 
        self.index = index 
        self.L = L 
        self.label2index_map = dict() 

    def partition_eval(self): 
        l_info = dict() 
        q = set(self.L) 

        for q_ in q: 
            l_info[q_] = list(self.index_info(q_))
            
        return l_info 

    """
    return: 
    - (min value,max value, mean value) of label along `index`
    """
    def index_info(self,l,store_indices:bool=True): 
        if l not in self.label2index_map: 
            indices = self.indices_of_label(l)
            assert len(indices) > 0 
            if store_indices: 
                self.label2index_map[l] = indices 
        else: 
            indices = self.label2index_map[l] 

        values = sorted([self.D_[i,self.index] for i in indices])
        min,max = values[0],values[-1] 
        ave = np.mean(values)
        return min,max,ave 

    def indices_of_label(self,l): 
        return np.where(self.L==l)[0] 