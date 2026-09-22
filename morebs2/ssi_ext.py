from .search_space_iterator import * 

"""
For a sequence |S| containing sequence sizes, iterator produces all 
|S|-indices. 

EX: (2,2,3) => 
(000),(100),(010),(110),(001),(101),(011),(111),(002),(102),(012),(112)
"""
class MultiIndexIterator: 

    def __init__(self,index_sequence): 
        assert type(index_sequence) in {list,np.ndarray} 
        index_sequence = np.array(index_sequence,dtype=int) 
        assert is_vector(index_sequence) 

        self.index_sequence = index_sequence  

        self.fin_stat = False 

        self.init_iterator() 
        return 

    def init_iterator(self): 
        bounds = np.zeros((len(self.index_sequence),2)) 
        bounds[:,1] = deepcopy(self.index_sequence) 

        startPoint = np.zeros((len(self.index_sequence),)) 
        columnOrder = np.arange(len(self.index_sequence)) 
        ssihop = deepcopy(self.index_sequence) 
        cycleOn = False 
        cycleIs = 0 

        self.ssi = SearchSpaceIterator(bounds,startPoint,columnOrder,\
            ssihop,cycleOn,cycleIs) 
        return

    def __next__(self): 
        if self.fin_stat: return None 

        q = next(self.ssi) 
        self.fin_stat = self.ssi.finished() 
        return q 


"""
Used to iterate through a sequence S, s.t. every element S' of S is a sequence. 
Produces all possible |S|-tuples, each |S|-tuple containing one element per S' of S.  
"""
class OrderedSelection(MultiIndexIterator):  

    def __init__(self,seq_of_seq): 
        assert type(seq_of_seq) == list 
        seq_of_seq = [list(s) for s in seq_of_seq] 

        self.sseq = seq_of_seq  

        index_seq = [len(s) for s in self.sseq] 

        super().__init__(index_seq)
        return 

    def __next__(self): 
        q = super().__next__() 
        if type(q) == type(None): return None 
        return [self.sseq[i][int(k)] for (i,k) in enumerate(q)] 