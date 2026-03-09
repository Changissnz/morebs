from .onedimprt_classifier import * 

"""
A memory-intensive extension to classifier<RecursiveOneDimClassifier>. 
Practically memorizes all (index,label) pairs for label vector L. 
With this information, classifier uses its root node to add the appropriate 
vector v_l to every input v, before classifier<RecursiveOneDimClassifier> 
proceeds with classifying (v_l + v). 

Due to this classifier's additive adjustment scheme relying on indices, 
the ordering of input samples matter for accuracy. In other words, a 
permutation D_ of the samples of dataset D will produce different output 
labels. This is a fundamental weakness of this memory-based schematic. 
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
        self.crepr = indexed_contiguous_repr__sequence(L)

        
    def make_additive_map(self,D,L):
        self.add_map = default_additive_map(D,L) 

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
    # NOTE: duplicate code from <OneDimClassifier> 
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

#------------------------------------------------------------------------------------------------


"""
Memory-Bound Alternating Additive Adjusted 
Recursive One-Dimensional Classifier 

This classifier works on the same principle as classifier<AdditiveAdjustedRecursiveODC>. 
However, it has additional parameters (`max_mem_sz`,`mem_ratio_per_node`) that limit the 
number of index-to-label ranges it can remember. 

Unlike classifier<AdditiveAdjustedRecursiveODC>, a structure that sets one node for additive adjustment
as the root, and then proceeds to classify the offsetted input vectors according to 
classifier<RecursiveOneDimClassifier>, every node of classifier<MBAAARecursiveODC> is of 
class<ODCAAPrefixedNode>. For an input vector v_i, this node adds a vector v_il, according to 
its index-to-label range information, to v_i, and then uses classifier<AdditiveAdjustedRecursiveODC> 
to classify (v_il + v_i). 

For sub-data D_ of D, and its associate index-to-label ranges R_, class<ODCAAPrefixedNode> can 
memorize at most 
    [ceil(`mem_ratio_per_node` * |R|),`max_mem_sz` - Q]; Q the number of ranges memorized by previous 
            nodes to the node instance. 
"""
class MBAAARecursiveODC(RecursiveOneDimClassifier): 

    def __init__(self,D,L,prg=None,pscheme=0,max_mem_sz=100,mem_ratio_per_node=0.5,\
        verbose:bool=False): 

        assert type(max_mem_sz) == int and max_mem_sz > 0 
        assert 0.0 < mem_ratio_per_node <= 1.0

        super().__init__(D,L,prg,pscheme,verbose) 


        self.max_mem_sz = max_mem_sz
        self.current_mem_sz = 0 
        self.mem_ratio_per_node = mem_ratio_per_node

        # every element is 
        # ODCNode
        # P := dict, label -> {indices of label in sub-data}
        # E := dict, label -> number of positive classification errors in D  
        # D := 2-d matrix, sub-data 
        # L := vector, labels corresponding to sub-data 
        return

    def init_root(self):
        self.root = ODCAAPrefixedNode(self.D,self.L,previous_indices=set()) 
        self.node_cache.append(self.root) 
        return

    def __next__(self): 
        if len(self.node_cache) == 0: 
            self.fin_stat = True 
        
        if self.fin_stat: 
            return False 

        node = self.node_cache.pop(0) 
        sz,child_nodes = node.make(self.max_mem_sz - self.current_mem_sz,\
            self.mem_ratio_per_node,self.prg,pscheme=self.pscheme)
        self.current_mem_sz += sz 
        self.node_cache.extend(child_nodes)

    """
    main method #1
    """
    def fit(self): 
        self.init_root() 
        while not self.fin_stat: 
            next(self) 
        self.clear_data() 

    """
    main method #2 
    """
    def classify(self,v): 
        q = self.root 
        l = None 
        self.sample_cls_path.clear() 
        while type(q) != type(None): 
            v,l,q = q.classify(v) 
            self.sample_cls_path.append((l,q))  
        return l 

    def clear_data(self): 
        self.D,self.L = None,None 