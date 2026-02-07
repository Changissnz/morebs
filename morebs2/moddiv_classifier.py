from .xclassif_aux import * 
from itertools import combinations 

def moddiv_op_on_vector(v,modulo,indices):
    v_ = v[[indices]]
    s = np.sum(v_) 
    return s % modulo, s // modulo 

class MultiDimModDivClassifierFunction:

    def __init__(self,modulo,index_set,label2moddiv_map): 
        assert type(index_set) == list 
        assert len(label2moddiv_map) > 0 

        self.modulo = modulo
        self.index_set = index_set 
        self.label2moddiv_map = label2moddiv_map
        self.sorted_labels = sorted(self.label2moddiv_map.keys()) 

    def __str__(self): 
        q = "modulo: ".format(self.modulo) + "\n" 
        q += "index set: ".format(self.index_set) + "\n" 
        for s in self.sorted_labels: 
            q += "* {}: {}\n".format(s,self.label2moddiv_map[s]) 
        return q 

    def classify(self,V): 
        m,d = moddiv_op_on_vector(V,self.modulo,self.index_set)
        return self.closest_label_to_moddiv(m,d) 

    def closest_label_to_moddiv(self,m,d): 
        q = [] 
        md = np.array([m,d])
        for l in self.sorted_labels: 
            mx = np.array(self.label2moddiv_map[l])
            q_ = euclidean_point_distance(mx,md)
            q.append((l,q_)) 
        return sorted(q,key=lambda x:x[1])[0][0]


class MultiDimModDivClassifier(XClassifier):

    def __init__(self,D,L,index_set,prg,attempts=20):  
        super().__init__(D,L,0) 
        assert type(prg) in {MethodType,FunctionType} 
        assert len(index_set) > 0 and type(index_set) == set 

        self.index_set = sorted(index_set)
        self.prg = prg 
        self.attempts = attempts 
        self.preproc() 
        self.init_candidates() 

        self.best_classifier = None 
        self.best_score = -float('inf') 
        return 

    def preproc(self): 
        self.l_info = dict() 
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
            #print("PAWNO")
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

        # calculate the modulo 
        s = 0 
        for v in self.l_info.values(): 
            s += v[q[0]][q[1]] 

        if q[1] == 0:
            a = -1 
        else:  
            a = 1 

        s += a 
        while s == 0: 
            s += a 

        # calculate the labels 
        lx = dict() 
        for l in self.L_: 
            m,d = self.moddiv_pair_for_label(l,s)
            lx[l] = (m,d) 
        return MultiDimModDivClassifierFunction(s,self.index_set,lx)

    def moddiv_pair_for_label(self,label,modulo): 
        indices = self.label2index_map[label] 
        assert len(indices) > 0 
        m,d = 0,0 

        for i in indices: 
            v = self.D_[i] 
            m_,d_ = moddiv_op_on_vector(v,modulo,self.index_set)
            m += m_ 
            d += d_ 
        return m / len(indices),d / len(indices)

# NOTE: careful with high dimensional datasets 
class ModDivClassifier:

    def __init__(self,D,L,prg,total_indexset_candidates=float('inf'),attempts_per=50,num_solutions=1):   
        assert type(prg) in {MethodType,FunctionType}

        self.D = D 
        self.L = L 
        self.prg = prg 
        self.total_indexset_candidates = total_indexset_candidates 
        self.attempts_per = attempts_per 
        self.num_solutions = num_solutions

        self.ref_dim = 1 
        self.next_combo_seq() 
        self.fin_stat = False 
        self.solutions = [] 

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
            if self.ref_dim > self.D.shape[1]: 
                self.fin_stat = True 
                return 
            self.ref_dim += 1 
            self.next_combo_seq() 
            return next(self)     

        self.total_indexset_candidates -= 1
        mdc = MultiDimModDivClassifier(self.D,self.L,q,self.prg,attempts=self.attempts_per)  
        mdc.make() 

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
