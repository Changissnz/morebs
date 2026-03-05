from .onedimprt_classifier import * 
from .seq_repr import contiguous_repr__sequence

"""
First node (root) for classifier<AdditiveAdjustedRecursiveODC>. 
Adds a vector v_l to input x, according to current index i that maps 
to vector v_l corresponding to a label l. 
"""
class ODCAdditiveAdjustmentNode: 

    def __init__(self,contiguous_sequence,additive_map):  
        self.contiguous_sequence = contiguous_sequence
        self.additive_map = additive_map 

        # counter variable
        self.c = 0
        self.index_counter = 0 
        self.part_index = 0 

    def map(self,v): 
        self.c += 1 
        self.index_counter += 1 

        q = self.contiguous_sequence[self.part_index] 
        # case: move to next 
        if self.index_counter > q[1]: 
           #print("NEXT PART FROM ",q) 
            self.index_counter = 1 
            self.part_index += 1 
            if self.part_index >= len(self.contiguous_sequence): 
                self.part_index = 0

        q = self.contiguous_sequence[self.part_index] 
        l = q[0] 
        v2 = self.additive_map[l] 
        return v + v2

    def reset(self): 
        self.c = 0 
        self.index_counter = 0 
        self.part_index = 0 

"""
A memory-intensive extension to classifier<RecursiveOneDimClassifier>. 
Practically memorizes all (index,label) pairs for label vector L. 
With this information, classifier uses its root node to add the appropriate 
vector v_l to every input v, before classifier<RecursiveOneDimClassifier> 
proceeds with classifying (v_l + v). 
"""
class AdditiveAdjustedRecursiveODC(RecursiveOneDimClassifier): 

    def __init__(self,D,L,prg=None,pscheme=0,verbose:bool=False): 
        D = self.preproc(D,L)
        super().__init__(D,L,prg,pscheme,verbose) 

    #--------------------------------------------------------------------

    def preproc(self,D,L): 
        self.make_additive_map(D,L) 
        self.make_adjustment(L) 
        self.root_ = ODCAdditiveAdjustmentNode(self.crepr,self.add_map) 

        D2 = np.array([self.root_.map(d) for d in D]) 
        self.root_.reset() 
        return D2 

    def make_adjustment(self,L): 
        self.crepr = contiguous_repr__sequence(L)
        
    def make_additive_map(self,D,L):
        labels = sorted(set(L)) 
        num_cols = D.shape[1]
        max_diff = max([max(D[:,i]) - min(D[:,i]) for i in range(num_cols)]) 
        increment = max_diff * 2  

        if increment == 0: 
            increment = 10 

        v = np.zeros((num_cols,)) 
        add_map = dict() 

        for (i,l) in enumerate(labels): 
            v_ = np.copy(v) 
            v_ +=  increment + increment * i * -1 ** i 
            add_map[l] = v_ 
        self.add_map = add_map 

    #----------------------------------------------------------------------

    """
    main method #2 
    """
    def classify(self,v): 
        v2 = self.root_.map(v) 
        return super().classify(v2) 

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

#------------------------------------------------------------------------------------------------
