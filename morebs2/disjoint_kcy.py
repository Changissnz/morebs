from .matrix_methods import * 
from copy import deepcopy

"""
Simultaneous iterator over K cycles. 

At every i'th timestamp, iterator outputs 

    <(v[j] + hop_info[j]) % cp_seq[j] : j in [0,|cp_seq| - 1].>

Iterator can also iterate in reverse. To change hop length, use method<update_hop>. 

NOTE: this is different from the iteration approach of class<SearchSpaceIterator>, which 
iterates by single dimension + possible carry-over. 
"""
class DisjointKCyclesIterator: 

    def __init__(self,cperiod_seq:list,hop=1):  
        
        if not is_number(hop,{float,\
            np.float16,np.float32,np.float64,\
            complex,np.complex64,np.complex128}): 
            
            hop = np.array(hop,dtype=int) 
            assert is_vector(hop) and len(hop) == len(cperiod_seq)
            self.hop_info = hop 
        else: 
            self.hop_info = np.ones((len(cperiod_seq),),dtype=int) * int(hop) 

        self.cp_seq = np.array(cperiod_seq,dtype=int)

        self.active_cycles = set([i for i in range(len(self.cp_seq))])

        # index counter -> cycle indices 
        self.index_map = defaultdict(set) 

        self.init_index() 

        self.rev_mode = False 
        return 

    def init_index(self,default_index=0): 
        self.index_map.clear() 
        
        for (i,h) in enumerate(self.hop_info): 
            self.index_map[default_index] |= {i} 
        return

    def current_index(self): 
        return deepcopy(self.index_map) 

    def update_hop(self,h,index=None): 
        h = int(h) 

        if type(index) == type(None): 
            self.hop_info = self.hop_info * 0 + h 
        else: 
            assert 0 <= index < len(self.hop_info)
            self.hop_info[index] = h 
        return 
        
    def __next__(self): 
        q = np.zeros((len(self.cp_seq),)) 

        new_index_map = defaultdict(set) 

        for k,v in self.index_map.items(): 

            for v_ in v: 
                clength = self.cp_seq[v_] 
                q[v_] = k % clength 

                if v_ not in self.active_cycles: 
                    new_index_map[k] |= {v_}  
                    continue 
                new_index_map[k+self.hop_info[v_]] |= {v_}

        self.index_map = new_index_map 
        self.rev_mode = False 

        return q 

    def reverse(self): 
        H = deepcopy(self.hop_info) 
        for (i,v) in enumerate(self.hop_info): 
            self.hop_info[i] = v * -1 

        q = deepcopy(self.rev_mode) 
        r = next(self) 
        self.rev_mode = q 

        if self.rev_mode: 
            r2 = r 
        else: 
            next(self) 
            r2 = next(self)

        self.hop_info = H 

        self.rev_mode = True 
        return r2 