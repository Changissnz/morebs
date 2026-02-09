from .xclassif_aux import * 
from itertools import combinations 
from math import floor 

def moddiv_op_on_vector(v,modulo,indices,W) :
    v_ = v[[indices]]

    if type(W) != type(None): 
        v_ = np.dot(v_,W)  
    s = np.sum(v_) 
    return s % modulo, s // modulo 

"""
l_info := dict, index -> label -> (min,max,mean) 
"""
def l_info_to_matrix(l_info,index_seq,element_index): 
    assert type(index_seq) == list 

    q = [] 

    labels = sorted(l_info[index_seq[0]].keys())
    for l in labels: 
        q_ = [] 
        for i in index_seq: 
            q_.append(l_info[i][l][element_index])  
        q.append(q_)
    return np.array(q)

class MultiDimModDivClassifierFunction:

    def __init__(self,modulo,index_set,label2moddiv_map,W=None): 
        assert type(index_set) == list 
        assert len(label2moddiv_map) > 0 

        self.modulo = modulo
        self.index_set = index_set 
        self.label2moddiv_map = label2moddiv_map
        self.W = W 
        self.sorted_labels = sorted(self.label2moddiv_map.keys()) 

    def __str__(self): 
        q = "modulo: {}".format(self.modulo) + "\n" 
        q += "index set: {}".format(self.index_set) + "\n" 
        for s in self.sorted_labels: 
            q += "* {}: {}\n".format(s,self.label2moddiv_map[s]) 
        return q 

    def classify(self,V): 
        m,d = moddiv_op_on_vector(V,self.modulo,self.index_set,self.W) 
        return self.closest_label_to_moddiv(m,d) 

    def closest_label_to_moddiv(self,m,d): 
        q = [] 
        md = np.array([m,d])
        for l in self.sorted_labels: 
            mx = np.array(self.label2moddiv_map[l])
            q_ = euclidean_point_distance(mx,md)
            q.append((l,q_)) 

        #print("YY: ",np.array([np.array(self.label2moddiv_map[l]) for l in self.sorted_labels]))
        x = sorted(q,key=lambda x:x[1])
        ##print("XX: ",x)
        return x[0][0]


class MultiDimModDivClassifier(XClassifier):

    def __init__(self,D,L,index_set,prg,attempts=20,l_info_=None,weighted=False,\
        verbose=False):  
        super().__init__(D,L,0) 
        assert type(prg) in {MethodType,FunctionType} 
        assert len(index_set) > 0 and type(index_set) == set 
        if type(l_info_) == tuple: 
            assert len(l_info_) == 3 
            assert type(l_info_[0]) == list 
            assert type(l_info_[1]) == dict 
        else: 
            assert type(l_info_) == type(None)

        self.index_set = sorted(index_set)
        self.prg = prg 
        self.attempts = attempts 

        # index -> label -> (min,max,mean)
        self.l_info = dict() 
        self.preproc(l_info_)
        self.weighted = weighted  
        self.verbose = verbose 
        self.init_candidates() 

        self.best_classifier = None 
        self.best_score = -float('inf') 
        return 

    def preproc(self,l_info_):
        if type(l_info_) != type(None): 
            self.L_ = l_info_[0] 
            self.l_info = l_info_[1] 
            self.label2index_map = l_info_[2] 
            return  

        for i in self.index_set: 
            self.index = i 
            self.l_info[i] = self.partition_eval() 
        self.L_ = sorted(set(self.L))

    def init_candidates(self): 
        self.candidates = [] 
        for k in self.L_: 
            for x in [0,1,2]: 
                self.candidates.append((k,x)) 
        return 

    def make(self): 
        while self.attempts > 0: 
            C = self.make_classifier_()
            if type(C) == type(None): break 
            q = self.score_accuracy(C)
            if q > self.best_score: 
                self.best_classifier = C 
                self.best_score = q 
            self.attempts -= 1

    def score_accuracy(self,C): 
        s = 0 
        for (i,v) in enumerate(self.D): 
            q = C.classify(v) 
            if q == self.L[i]: 
                s += 1 
        return s 

    def choose_label_candidate(self): 
        if len(self.candidates) == 0: 
            self.fin_stat = True 
            return 

        i = int(self.prg()) % len(self.candidates) 
        return self.candidates.pop(i)

    def make_classifier_(self): 
        q = self.choose_label_candidate() 
        if type(q) == type(None): 
            return 

        # case: weighted 
        if self.weighted: 
            M = l_info_to_matrix(self.l_info,self.index_set,q[1])
            otl = OneTenLinearFunctionDifferenceMaximizer(M,alter_pattern="linear")
            otl.search() 
            W = otl.best_W
            if self.verbose: print("best weight: ",W) 
        else: 
            W = None 
        s = self.calculate_modulo(q,W) 
        if self.verbose: print("modulo: ",s) 
        # calculate the labels 
        lx = dict() 
        for l in self.L_: 
            m,d = self.moddiv_pair_for_label(l,s,W)  
            lx[l] = (m,d) 

        return MultiDimModDivClassifierFunction(s,self.index_set,lx,W) 

    def calculate_modulo(self,li,W=None): 
        # calculate the modulo 
        s = [] 

        indices = sorted(self.l_info.keys())
        for i in indices: 
            v = self.l_info[i] 
            s.append(v[li[0]][li[1]])
        s = np.array(s) 
        if type(W) != type(None): 
            s = np.dot(s,W) 
        else: 
            s = np.sum(s) 

        if li[1] == 0:
            a = -1 
        else:  
            a = 1 

        s += a 
        while s == 0: 
            s += a 
        return s 

    def moddiv_pair_for_label(self,label,modulo,W=None): 
        indices = self.label2index_map[label] 
        assert len(indices) > 0 
        m,d = 0,0 

        for i in indices: 
            v = self.D_[i] 
            m_,d_ = moddiv_op_on_vector(v,modulo,self.index_set,W)
            m += m_ 
            d += d_ 
        return m / len(indices),d / len(indices)

# NOTE: careful with high dimensional datasets 
# NOTE: setting mode `weighted` to True does not seem to improve performance on some datasets.
#               bug OR limit of algorithm?
class ModDivClassifier:

    def __init__(self,D,L,prg,index_size_search_direction="min",total_indexset_candidates=float('inf'),attempts_per=50,num_solutions=1,weighted=False,verbose=False):   
        assert type(prg) in {MethodType,FunctionType}
        assert index_size_search_direction in {"min","max","mean"}

        self.D = D 
        self.L = L 
        self.prg = prg 
        self.index_size_sdir = index_size_search_direction
        self.total_indexset_candidates = total_indexset_candidates 
        self.attempts_per = attempts_per 
        self.weighted = weighted 
        self.verbose = verbose 
        self.num_solutions = num_solutions

        self.ref_dim = None 
        self.init_index_size_search_cache()
        self.next_combo_seq() 
        self.fin_stat = False 
        self.solutions = [] 

        # list<labels>,index -> label -> (min,max,mean)
        self.l_info_ = None 
        self.init_label_info_map() 

    def switch_next_mode(self,stat): 
        assert type(stat) == bool 
        self.return_classifier = stat 

    def init_label_info_map(self): 
        q = set([i for i in range(self.D.shape[1])]) 
        mdc = MultiDimModDivClassifier(self.D,self.L,q,self.prg,attempts=self.attempts_per,l_info_=self.l_info_,weighted=self.weighted)
        self.l_info_ = (mdc.L_,mdc.l_info,mdc.label2index_map) 

    def init_index_size_search_cache(self): 
        self.ref_candidates = [i for i in range(1,self.D.shape[1]+1)] 
        self.next_ref_dim() 

    def next_ref_dim(self): 
        self.ref_dim = None 
        if len(self.ref_candidates) == 0: 
            return 

        if self.index_size_sdir == "min": 
            self.ref_dim = self.ref_candidates.pop(0) 
        elif self.index_size_sdir == "max": 
            self.ref_dim = self.ref_candidates.pop(-1) 
        else: 
            m = floor(len(self.ref_candidates) / 2) 
            self.ref_dim = self.ref_candidates.pop(m)  
        return 

    def next_combo_seq(self): 
        self.combos = combinations([i for i in range(self.D.shape[1])],self.ref_dim) 

    def make(self): 
        while not self.fin_stat: 
            next(self) 

    def __next__(self): 
        if self.fin_stat: return 

        if self.total_indexset_candidates == 0: 
            self.fin_stat = True 
            return 

        try:
            q = set(next(self.combos))
        except: 
            self.next_ref_dim() 
            if type(self.ref_dim) == type(None): 
                self.fin_stat = True 
                return 

            self.next_combo_seq() 
            return next(self)     

        self.total_indexset_candidates -= 1
        if self.verbose: 
            print("* one classifier with index subset: {}".format(q)) 
        
        q_linfo = {q_:self.l_info_[1][q_] for q_ in q} 
        q_linfo = (self.l_info_[0],q_linfo,self.l_info_[2])
        mdc = MultiDimModDivClassifier(self.D,self.L,q,self.prg,attempts=self.attempts_per,\
            l_info_=q_linfo,weighted=self.weighted,verbose=self.verbose)  
        mdc.make() 
        self.log_new_solution(mdc) 

    def log_new_solution(self,mdc): 
        if len(self.solutions) < self.num_solutions: 
            self.solutions.append((mdc.best_classifier,mdc.best_score)) 
            self.solutions = sorted(self.solutions,key=lambda x:x[1],reverse=True) 
        else: 
            for (i,x) in enumerate(self.solutions): 
                if mdc.best_score > x[1]: 
                    self.solutions.insert(i,(mdc.best_classifier,mdc.best_score))
                    self.solutions = self.solutions[:self.num_solutions] 
                    break 
        return 
