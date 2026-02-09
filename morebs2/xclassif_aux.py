from .numerical_generator import * 
from .search_space_iterator import * 
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

#---------------------------------------------------------------------------------

def cumulative_vector_difference(V,add_log:bool=False): 
    S = 0 
    for i in range(len(V) - 1): 
        q = V[i] 
        for j in range(i+1,len(V)): 
            q2 = abs(q-V[j]) 
            if add_log: 
                q2 = abs(safe_log(q2,0)) 
            S += q2 
    return S 

# NOTE: careful with large vector values 
class OneTenLinearFunctionDifferenceMaximizer:

    def __init__(self,samples,alter_pattern="linear",verbose=False):
        assert is_2dmatrix(samples) 
        assert alter_pattern in {"linear","combinative"}  

        self.samples = samples 
        self.alter_pattern = alter_pattern 
        self.verbose = verbose 

        self.W = -np.ones((self.samples.shape[1],)) 
        self.index = 0 

        self.best_W = None 
        self.best_score = -float('inf') 

        q = np.zeros((self.samples.shape[1],2))
        #q[:,0],q[:,1] = 1,19 
        q[:,0],q[:,1] = -1,3 

        self.set_index_ordering()
        self.ssi = SearchSpaceIterator(q,deepcopy(self.W),\
            self.ordering,SSIHop = 2,cycleOn = False, cycleIs = 0) 
        self.fin_stat = False 
        return 

    def set_index_ordering(self): 
        diff_vec = [(i,cumulative_vector_difference(self.samples[:,i],True)) for i in range(self.samples.shape[1])]
        diff_vec = sorted(diff_vec,key=lambda x:x[1]) 
        self.ordering = [d[0] for d in diff_vec]
        return 

    def search(self): 
        while not self.fin_stat: 
            next(self) 
        
        for i in range(len(self.best_W)): 
            if self.best_W[i] == 1:
                self.best_W[i] = 10 

    def __next__(self): 
        if self.fin_stat: return 

        q = self.next_element() 
        if type(q) == type(None): 
            self.fin_stat = True 
            return 
        
        V = np.dot(self.samples,q) 
        diff = cumulative_vector_difference(V,True) 

        if self.verbose: 
            print("* V: {}\n* score: {}".format(q,diff /1000)) 

        if diff > self.best_score: 
            self.best_W = deepcopy(q) 
            self.best_score = diff 

    def next_element(self): 
        if self.alter_pattern == "linear": 
            if self.index >=len(self.W): 
                return None
            W_ = deepcopy(self.W) 
            index = self.ordering[self.index] 
            self.W[index] = 10 
            self.index += 1 
            return W_ 
        else: 
            if self.ssi.finished(): 
                return None 
            self.W = next(self.ssi) 
            return self.W 