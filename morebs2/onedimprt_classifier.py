from .odc_nodes import * 

"""
A dataset classifier, based on decision tree format. 
Uses the classifier <OneDimClassifier> as a unit node. 
"""
class RecursiveOneDimClassifier: 

    def __init__(self,D,L,prg=None,pscheme=0,verbose:bool=False): 
        assert is_2dmatrix(D) 
        assert is_vector(L) 
        self.D = D 
        self.L = L 
        self.fin_stat = False 

        if type(prg) not in {MethodType,FunctionType}: 
            self.prg = default_std_Python_prng()
        else: 
            self.prg = prg 
        assert pscheme in {0,1} or type(pscheme) in {MethodType,FunctionType}
        self.pscheme = pscheme 

        assert type(verbose) == bool 
        self.verbose = verbose 
        self.root = None 

        # every element is 
        # [0] ODCNode
        # [1] P := dict, label -> {indices of label in sub-data}
        # [2] E := dict, label -> number of positive classification errors in D  
        # [3] D := 2-d matrix, sub-data 
        # [4] L := vector, labels corresponding to sub-data 
        self.node_cache = [] 
        self.current = None 

        self.sample_cls_path = [] 

    def clear_data(self): 
        self.D = None 
        self.L = None 

    #--------------------------- main methods 

    """
    main method #3 
    """
    def score_accuracy(self,D,L): 
        assert len(D) == len(L) 

        c = 0 
        for (x,l) in zip(D,L): 
            l_ = self.classify(x) 
            if l_ == l: 
                c += 1 
        return c 

    """
    main method #2 
    """
    def classify(self,v): 
        q = self.root 
        l = None 
        self.sample_cls_path.clear() 
        while type(q) != type(None): 
            l,q = q.classify(v) 
            self.sample_cls_path.append((l,q))  
        return l 

    #------------------------------------- used to find classification solution 

    """
    main method #1 
    """
    def fit(self): 
        self.init_root() 
        while not self.fin_stat: 
            self.next_process() 
        self.clear_data() 

    def init_root(self): 
        self.one_process(self.D,self.L)  
        self.root = self.node_cache[0][0] 

    def next_process(self): 
        if self.fin_stat: 
            return 

        # case: load a new node of interest 
        if type(self.current) == type(None): 
            if len(self.node_cache) == 0: 
                self.fin_stat = True 
                return 
            self.current = self.node_cache.pop(0) 

        # iterate through and clear out any labels with 0 error s
        keys = sorted(self.current[2].keys())
        key0 = [] 
        for k in keys: 
            if len(self.current[2][k]) == 0:
                del self.current[2][k] 
                key0.append(k) 
        rem_keys = sorted(set(keys) - set(key0)) 

        # case: perfect labeling 
        if len(rem_keys) == 0: 
            if self.verbose: 
                print("\t\t** one node")
                print(self.current[0]) 
            self.current = None 
            return 

        i = prg__single_to_int(self.prg)() % len(rem_keys) 
        subset_label = rem_keys[i] 
        del self.current[2][subset_label]

        D,L = [],[]
        for i in self.current[1][subset_label]: 
            D.append(self.current[3][i]) 
            L.append(self.current[4][i]) 
        D,L = np.array(D),np.array(L)
        odcn2,P2,E2 = self.one_child(self.current[0],subset_label,D,L)

    def one_process(self,D,L,previous_indices=set()): 
        odc,P,E = self.one_classification_process_(D,L,previous_indices)
        if type(odc) == type(None): 
            return None,None,None 

        self.node_cache.append((odc,P,E,D,L)) 
        return odc,P,E 

    """
    return: 
    [0] ODCNode 
    [1] dict, label -> {indices}
    [2] dict, label -> {mislabeled indices} 
    """ 
    def one_classification_process_(self,D,L,previous_indices=set()):  
        # instantiate one classifier 
        all_cols = set([i for i in range(D.shape[1])]) 
        candidates = sorted(all_cols - previous_indices) 
        if len(candidates) == 0: 
            if self.verbose: print("? no more indices")
            return None,None,None 

        i = prg__single_to_int(self.prg)() % len(candidates)  
        index = candidates[i] 
        odc = OneDimClassifier(D,L,index,self.pscheme)
        odc.make() 

        # partition by classification 
        P = defaultdict(set) 
        E = defaultdict(set)#int)
        for i,x in enumerate(D): 
            l = odc.classify(x) 
            P[l] |= {i} 
            if l != L[i]: 
                #E[l] += 1
                E[l] |= {i} 
        odcn = ODCNode(odc.clf,dict(),previous_indices | {index}) 
        return odcn,P,E 

    def one_child(self,odcn,subset_label,D,L): 
        previous_indices = odcn.previous_indices
        odcn2,P,E = self.one_process(D,L,previous_indices)
        if type(odcn2) ==type(None): return None,None,None 
        odcn.add_nextnode(odcn2,subset_label) 
        return odcn2,P,E 
