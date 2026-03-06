from morebs2.deline import *

class DPointAnalysis:
    '''
    provides a description of the complexity of the dataset
    of two-dimensional labelled points that :class:`DLineateMC`
    delineates. 
    '''

    def __init__(self):
        return

"""
Node container for <DLineate22> instance. Used to form trees, 
and these trees are used by class<DLineateMC> for classification. 
"""
class D22Node: 

    def __init__(self,d22,child_nodes):  
        assert type(d22) == DLineate22
        assert type(child_nodes) == list 
        q = []
        for c in child_nodes: 
            assert type(c) == DLineate22
            assert c.parent_idn() == d22.idn 
            q.append(c.idn)
        assert d22.children_idn() == q

        self.d22 = d22
        self.child_nodes = child_nodes

    def idn(self): 
        return self.d22.idn 

    def parent_idn(self): 
        return self.d22.parent_idn() 

    def children_idn(self): 
        return self.d22.children_idn() 

    def add_child_node(self,c): 
        assert type(c) == D22Node 
        assert c.parent_idn() == self.idn() 
        self.child_nodes.append(c) 

    def classify_point(self,x): 
        return self.d22.classify_point(x) 
    
    def classify(self,x): 

        c = self.classify_point(x)
        if c == -1: 
            return -1 

        q = [c] 
        nodes = [self] 
        while len(nodes) > 0: 
            node = nodes.pop(0) 
            cs = [] 

            for cn in node.child_nodes: 
                c2 = cn.classify_point(x) 
                if c2 != -1: 
                    cs.append(c2) 
                    nodes.append(cn)
            q.extend(cs) 

        return q

    def node_count(self): 
        c = 1 
        nodes = [self]

        while len(nodes) > 0: 
            node = nodes.pop(0)
            x = len(node.child_nodes) 
            c += x 
            nodes.extend(node.child_nodes) 
        return c 


def most_frequent_label(V):
    d = Counter() 
    for v_ in V: 
        d[v_] += 1 
    d = [(k,v) for k,v in d.items()] 
    return max(d,key=lambda x: x[1])[0] 


"""
A multi-classifier for 2-dimensional dataset built on top of class<DLineate22>. 
"""
class DLineateMC:
    '''
    NOTE: inefficiencies exist. Runtime "slow" for > 200 samples. 
    '''

    def __init__(self,xyl,dmethod="nodup",prg=prg__LCG(-1615.4,-3454.66,-919.16,-9191.4545)):
        self.xyl = xyl
        self.dmethod = dmethod
        self.prg = prg 
        self.xyl_unproc = deepcopy(self.xyl)
        self.d = None
        self.dds = []
        self.didn = "0"

        self.done_cache = [] 
        self.node_cache = [] 
        self.child_cache = [] 
        self.NOT_cache = [] 

        self.fin_stat = False 
        self.cl = None 

    #------------------------------------------------------------------------------------------

    def init_root(self,clockwise=True,max_points_per_edge=100):
        '''
        '''

        self.d = DLineate22(deepcopy(self.xyl_unproc),clockwise=clockwise,dmethod = self.dmethod,\
            idn=self.didn,target_min_label=False,max_points_per_edge=max_points_per_edge,\
            prg=self.prg)

        self.increment_didn()
        _,_,_,met = self.d.full_process(None)
        self.node_cache.append((self.d,met)) 
        return

    def __next__(self):
        if self.fin_stat: 
            return False 

        if len(self.node_cache) == 0: 
            self.fin_stat = True 
            return False 

        parent_d22,parent_met = self.node_cache.pop(0) 
        self.process_D22(parent_d22,parent_met) 
        return True 

    def process_D22(self,parent_d22,parent_met,clockwise=True):
        D_fpos,met_fpos = self.process_D22_(parent_d22,parent_met,clockwise,process_type="fpos")
        
        if type(D_fpos) != type(None): 
            D_fpos,met_fpos = self.post_process_d22(parent_d22,D_fpos,met_fpos)
            self.node_cache.append((D_fpos,met_fpos)) 

        self.process_D22_remaining_NOTs(parent_d22,parent_met,clockwise)
        self.done_cache.append(parent_d22) 

        '''
        self.idn = d22idn 
        self.label = label
        self.num_elements = num_elements 
        self.contained_indices = contained_indices
        self.fpos_indices = false_pos_indices
        ''' 
        return -1 

    def process_D22_(self,parent_d22,parent_met,clockwise=True,process_type="fpos",not_indices=None): 
        assert process_type in {'fpos','not'}
        # process first child 

        if process_type == "fpos": 
            xyl = parent_d22.xyl[parent_met.fpos_indices] 
        else: 
            if type(not_indices) == type(None): 
                not_indices = parent_met.NOT_indices() 
            
            xyl = parent_d22.xyl[not_indices] 

        if len(xyl) == 0: 
            return None,None 

        D = DLineate22(xyl,clockwise=clockwise,dmethod = self.dmethod,\
            idn=self.didn,target_min_label=False,max_points_per_edge=100,\
            prg=self.prg)
        self.increment_didn() 

        idn = None if process_type == "not" else parent_d22.idn
        _,_,_,met2 = D.full_process(idn,xyl) 
        return D,met2

    def process_D22_remaining_NOTs(self,parent_d22,parent_met,clockwise=True): 

        met_seq = [parent_met]
        d22_seq = [parent_d22] 

        not_indices = parent_met.NOT_indices()

        def fetch_not_contained_indices(ref_met): 
            NOT = set(ref_met.NOT_indices())
            for x in met_seq: 
                not2 = set(x.NOT_indices())
                NOT = NOT.intersection(not2) 
            return sorted(NOT) 

        while len(not_indices) > 0: 
            D,met2 = self.process_D22_(parent_d22,parent_met,clockwise,"not",not_indices)   
            if type(D) == type(None): continue 

            not_indices = fetch_not_contained_indices(met2) 
            d22_seq.append(D)
            met_seq.append(met2) 

        met_seq.pop(0) 
        d22_seq.pop(0) 

        while len(d22_seq) > 0: 
            d,m = d22_seq.pop(0),met_seq.pop(0) 
            d,m = self.post_process_d22(parent_d22,d,m)
            self.node_cache.append((d,m))
        return

    # NOTE: function does nothing
    def post_process_d22(self,parent_d22,d22,d22_metric): 
        """
        print("PP ",parent_d22.idn,d22.idn) 
        print("DXDX")
        print(parent_d22.xyl)
        print("dxdx")
        print(d22.xyl)
        print("D22 METRIC")
        print(d22_metric)
        data_IN = parent_d22.xyl[d22_metric.contained_indices]
        _,_,_,d22_metric = d22.analyze_delineation(data_IN) 
        d22.xyl = data_IN 

        print("DATA IS")
        print(data_IN) 
        """
        return d22,d22_metric 

    def increment_didn(self):
        self.didn = str(int(self.didn) + 1)

    def decrement_didn(self):
        self.didn = str(int(self.didn) - 1)

    def to_graph(self):
        x = {}
        for q in self.done_cache: 
            x[q.idn] = D22Node(q,[])
        
        S = [] 

        for q in self.done_cache: 
            p = q.parent_idn()
            if type(p) == type(None): 
                S.append(x[q.idn]) 
                continue 
            x[p].add_child_node(x[q.idn])
        self.cl = S 
        return

    def classify_(self,x): 
        cmap = {} 
        for q in self.cl: 
            c = q.classify(x)
            cmap[q.idn()] = c 
        return cmap 

    def classify(self,x): 
        cmap = self.classify_(x) 
        C = [] 
        for k,v in cmap.items(): 
            if v == -1: 
                continue
            q = v[-1]
            C.append(q)  

        if len(C) == 0: return -1 
        return sorted(C) 

    def one_classification(self,x): 
        v = self.classify(x) 
        if v == -1: return -1 
        return most_frequent_label(v)

    def full_run(self): 
        self.init_root() 
        stat = True 
        while stat: 
            stat = next(self) 
        self.to_graph() 

    def node_count(self): 
        cmap = {} 
        for q in self.cl: 
            cmap[q.idn()] = q.node_count() 
        return cmap 

DEFAULT_D22MC_MAX_POINTS_PER_LABEL = 20 

"""
An approximation built on top of class<DLineateMC>. By default, takes at most 20 
points per label of two-dimensional dataset. 
"""
class DLineateMCApprox(DLineateMC): 

    def __init__(self,xyl,dmethod="nodup",prg=prg__LCG(-1615.4,-3454.66,-919.16,-9191.4545)):
        self.D = xyl 
        xyl = approximate_points_for_delineation(self.D,\
            max_points_per_label=DEFAULT_D22MC_MAX_POINTS_PER_LABEL,prg=prg) 
        super().__init__(xyl,dmethod,prg)        
        return 

"""
Uses m <= `max_approximators` approximators to find a multi-class classifier. 
Built on top of class<DLineateMCApprox>. 
"""
class DLineateMCApproximators: 

    def __init__(self,xyl,max_approximators,dmethod="nodup",\
        prg=prg__LCG(-1615.4,-3454.66,-919.16,-9191.4545)):
        assert type(max_approximators) == int and max_approximators > 0 

        self.xyl = xyl 
        self.rem_xyl = deepcopy(xyl)
        self.max_approximators = max_approximators
        self.dmethod = dmethod 
        self.prg = prg 
        self.approximators = [] 
        self.fin_stat = False 

    def full_run(self): 
        while not self.fin_stat: 
            next(self) 

    def __next__(self): 
        if self.fin_stat: 
            return False 

        if len(self.approximators) == self.max_approximators: 
            self.fin_stat = True 
            return False 
        
        self.one_approximator()
        self.gauge_approximator() 

    def one_approximator(self): 
        dma = DLineateMCApprox(self.rem_xyl,dmethod=self.dmethod,prg=self.prg)
        dma.full_run() 
        self.approximators.append(dma) 

    def gauge_approximator(self): 
        dma = self.approximators[-1] 
        indices = [] 
        for i,x in enumerate(self.rem_xyl): 
            l_ = dma.one_classification(x[:2])
            if l_ == -1: 
                indices.append(i)
        if len(indices) == 0: 
            self.fin_stat = True 
            return 

        self.rem_xyl = self.rem_xyl[indices]

    def one_classification(self,x): 
        l = [] 
        for a in self.approximators: 
            l_ = a.one_classification(x)
            if l_ == -1: continue 
            l.append(l_) 

        if len(l) == 0: return -1 
        return most_frequent_label(l) 
