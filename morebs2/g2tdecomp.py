"""
graph-to-tree decomposition
"""
from .graph_basics import * 
from .numerical_generator import prg_seqsort,prg_seqsort_ties

class TNode: 

    def __init__(self,idn,next_cycling:bool=False,root_stat:bool=False,\
        rdistance:int = 0): 
        self.idn = idn 
        self.next_cycling = next_cycling
        self.children = []  
        self.xc_idn = set() 
        self.cindex = 0 # used to traverse children node in search process 
        self.root_stat = root_stat 
        self.scached = False 
        self.rdistance = rdistance 

        self.xclist = [] 
        return

    @staticmethod
    def dfs(tn,display:bool,collect:bool,reset_index:bool):
        if not display and not collect: return 
        assert type(tn) == TNode 
        if reset_index: tn.cindex = 0 
        cache = [tn] 
        d = defaultdict(set)
        mdepth = 0  
        while len(cache) > 0: 
            t_ = cache.pop(0)
            if display: 
                print(t_)
            cx = set([c.idn for c in t_.children])

            if collect: 
                d[t_.idn] = d[t_.idn] | cx 
            
            q = next(t_)
            if type(q) != type(None): 
                if reset_index: 
                    q.cindex = 0 
                cache.insert(0,t_)
                cache.insert(0,q) 
                mdepth = max([q.rdistance,mdepth])
            else: 
                if reset_index:
                    t_.cindex = 0 
        return d,mdepth 

    def index_of_child(self,idn): 
        for (i,c) in enumerate(self.children): 
            if c.idn == idn: 
                return i 
        return -1 

    def add_children(self,cs): 
        for c in cs: 
            assert type(c) == TNode 
            if c.idn in self.xc_idn: continue 
            self.children.append(c)

    def add_xclusion(self,xclude):
        assert type(xclude) == set 
        self.xclist.extend(xclude) 

    def __next__(self):
        if not self.next_cycling and self.cindex >= len(self.children): 
            return None 
        if len(self.children) == 0: return None 
        
        q = self.children[self.cindex % len(self.children)] 
        self.cindex += 1
        return q

    def __str__(self): 
        s = "idn:\t" + str(self.idn) + "\n"
        s += "rdistance:\t" + str(self.rdistance) + "  index:\t" \
            + str(self.cindex) + "\n"
        q = [str(c.idn) for c in self.children] 
        q = " ".join(q) 
        s += "children: " + q + "\n"
        s += "xchildren: " + str(self.xc_idn) + "\n"
        s += "is root: " + str(self.root_stat) + "\n"
        return s 

class G2TDecomp: 

    def __init__(self,d,decomp_rootnodes=[],excl_mem_depth=-1,\
            child_capacity=float('inf'),parent_capacity=float('inf'),prg=None): 

        assert type(d) == defaultdict
        assert d.default_factory == set 
        assert child_capacity > 0 and parent_capacity > 0

        graph_childkey_fillin(d)
        self.d = d 
        self.d_ = deepcopy(self.d)
        self.is_directed = is_directed_graph(d) 
        self.rn = decomp_rootnodes
        self.dr_map = defaultdict(int) 
        self.excl_mem_depth = excl_mem_depth
        self.cc = child_capacity
        self.pc = parent_capacity
        self.prg = None  
        self.predecomp() 

        # vars used for dfs search 
        self.decomp_queue = [] 
            # stores skipped nodes for every dfs search 
            # by a root node 
        self.skipped_nodes = [] 
            # neighbor-parent degree map. used as reference to 
            # satisfy upper-threshold values set by `excl_mem_depth`,
            # `cc`, and `pc`. 
        self.cdeg_map = None 
        self.cdeg_map2 = None 

    def predecomp(self):
        if len(self.rn) > 0: 
            return 
        
        gd = GraphComponentDecomposition(self.d) 
        gd.decompose() 
        self.dr_map = gd.depth_rank_map() 
        self.rn = [(k,v) for k,v in self.dr_map.items()] 

        if type(self.prg) == type(None): 
            self.rn = sorted(self.rn,key=lambda x:x[1]) 
        else: 
            vf = lambda x: x[1] 
            self.rn = prg_seqsort_ties(self.rn,self.prg,vf)
        return

    def next_key(self): 
        '''
        initializes a new tree, represented by class<TNode>. 
        '''
        if len(self.rn) == 0: 
            return False 

        x = self.rn.pop(0)
        tn = TNode(x,False,True,0)
        self.decomp_queue.append(tn) 
        self.store_np_degrees() 

    #--------------------- conditional methods for next 

    def store_np_degrees(self): 
        q = np_degree_map(self.d_)
        self.cdeg_map = defaultdict(int) 
        for k,v in q.items(): 
            self.cdeg_map[k] += v[0] + v[1] 
        self.cdeg_map2 = deepcopy(self.cdeg_map)
        return


