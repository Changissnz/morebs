"""
graph-to-tree decomposition
"""
from collections import defaultdict 
from copy import deepcopy 

def is_directed_graph(d):
    assert type(d) in {defaultdict,dict}
    
    for k,v in d.items(): 
        for v_ in v: 
            if v_ not in d: return False 
            if k not in d[v_]: return False 
    return True 

def prg_seqsort(l,prg): 
    l_ = [] 
    while len(l) > 0: 
        i = prg() % len(l) 
        l_.append(l.pop(i))
    return l_ 

class TNode: 

    def __init__(self,idn,next_cycling:bool=False,root_stat:bool=False,\
        rdistance:int = 0): 
        self.idn = idn 
        self.next_cycling = next_cycling
        self.children = []  
        self.cindex = 0 # used to traverse children node in search process 
        self.root_stat = root_stat 
        self.scached = False 
        self.rdistance = rdistance 
        return

    @staticmethod
    def dfs(tn,display:bool,collect:bool,reset_index:bool):
        if not display and not collect: return 
        assert type(tn) == TNode 
        if reset_index: tn.cindex = 0 
        cache = [tn] 
        d = defaultdict(set) 
        while len(cache) > 0: 
            t_ = cache.pop(0)
            if display: 
                print(t_)
            cx = set([c.idn for c in t_.children])

            if collect: 
                d[t_] = d[t_] | cx 
            
            q = next(t_)
            if type(q) != type(None): 
                if reset_index: 
                    q.cindex = 0 
                cache.insert(0,t_)
                cache.insert(0,q) 
        return d  

    def index_of_child(self,idn): 
        for (i,c) in enumerate(self.children): 
            if c.idn == idn: 
                return i 
        return -1 

    def add_children(self,cs): 
        for c in cs: assert type(c) == TNode 
        self.children.extend(cs) 

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
        s += "is root: " + str(self.root_stat) + "\n"
        return s 


class G2TDecomp: 

    def __init__(self,d,decomp_type:int,\
        decomp_rootnodes=[],prg=None):
        assert decomp_type in {1,2}
        assert type(d) == defaultdict 
        self.d = d
        self.d_ = deepcopy(self.d)
        self.untouched_nodes = set(self.d.keys()) 
        for v in self.d.values():
            self.untouched_nodes |= set(v) 

        self.dtype = decomp_type  
        self.decomp_rootnodes = decomp_rootnodes
        self.prg = prg 

        # a collection of tree instances 
        self.decompositions = dict() 
        self.decomp = []
        self.components = dict() # of sets 
        self.component = set() 

        self.dfs_nodecache = [] 
        self.skipped_edges = [] 

    def decompose(self): 
        stat = True 
        while stat: 
            stat = self.tdecomp_one_component() 

    def tdecomp_one_component(self): 
        stat = self.init_component_search()
        if not stat: return False 
        
        # half 1 
        while len(self.dfs_nodecache) > 0: 
            tn = self.dfs_nodecache.pop(0)
            self.next_from_TNode(tn)

        # half 2 
        stat = True 
        while stat: 
            stat = self.tree_from_skipped_edge() 

        if len(self.decompositions) > 0: 
            k = max(self.decompositions) + 1 
        else: 
            k = 0 

        self.decompositions[k] = self.decomp 
        self.decomp = [] 

        i = 0
        while i < len(self.decomp_rootnodes): 
            x = self.decomp_rootnodes[i] 
            if x in self.component: 
                self.decomp_rootnodes.pop(i) 
            else: 
                i += 1 

        self.components[k] = self.component 
        self.component = set() 
        return True 

    def connected_to(self,idn): 
        neighbors = deepcopy(self.d_[idn])
        for k,v in self.d_.items():
            if idn in v: 
                neighbors |= {k}
        return neighbors 

    #----------------------- first half of tree decomposition for 
    #----------------------- component 

    def init_component_search_(self): 
        assert len(self.dfs_nodecache) == 0 
        self.skipped_edges.clear() 

        if len(self.decomp_rootnodes) == 0: 
            if len(self.untouched_nodes) == 0: 
                return None 
            return self.untouched_nodes.pop()
        else: 
            return self.decomp_rootnodes.pop(0) 

    def init_component_search(self): 
        x = self.init_component_search_()
        if type(x) == type(None): 
            return False 

        tn = TNode(x,root_stat=True,rdistance=0) 
        self.decomp = [tn] 
        self.dfs_nodecache.append(tn) 
        self.untouched_nodes -= {x}
        self.component |= {x} 
        return True 
        
    """
    sets children for new <TNode>
    """
    def next_from_TNode_p1(self,tn): 
        if len(tn.children) > 0: return 

        if type(self.prg) != type(None): 
            self.prg_next_selector(tn) 
        else: 
            self.default_next_selector(tn) 

    def next_from_TNode_p2(self,tn):
        q = next(tn) 
        if type(q) == type(None): 
            return False 
        self.dfs_nodecache.insert(0,tn)
        self.dfs_nodecache.insert(0,q) 
        return True 

    def next_from_TNode(self,tn): 
        assert type(tn) == TNode 
        self.next_from_TNode_p1(tn) 
        return self.next_from_TNode_p2(tn)

    ################### next node selectors for tree construction 

    def prg_next_selector(self,tn): 
        neighbors = self.connected_to(tn.idn)
        if len(neighbors) == 0: return False 

        c = [] 
        skipped_nodes = set()
        self.component |= neighbors
        for x in neighbors: 
            # case: parental neighbor 
            if x not in self.d_[self.tn.idn]: 
                self.remove_edge(x,tn.idn,False)
                skipped_nodes |= {x} 
                continue 

            stat = (self.d_[x] - {tn.idn}) == 0 
            self.remove_edge(tn.idn,x,False)

            skipped = 1
            # case: x does not have any other neighbors 
            #       besides from `tn.idn`. Add it.  
            if stat: 
                c.append(TNode(x,rdistance=tn.rdistance+1)) 
                self.remove_edge(x,tn.idn,False)
                skipped = 0  
            # case: prg selects x to be a next 
            elif self.prg() % 2: 
                c.append(TNode(x,rdistance=tn.rdistance+1)) 
                self.remove_edge(x,tn.idn,False)
                skipped = 0 

            # case: skipped child after add-child decision
            if skipped: 
                skipped_nodes |= {x} 
            else:
                self.untouched_nodes -= {x}
        tn.add_children(c)
        self.add_skipped_edges(tn.idn,skipped_nodes)
        return True  

    def default_next_selector(self,tn): 
        neighbors = self.connected_to(tn.idn)
        if len(neighbors) == 0: return False 
        self.component |= neighbors

        c = []
        skipped_nodes = set()  
        for x in neighbors: 
            # case: parental neighbor, skip it 
            if x not in self.d_[tn.idn]: 
                skipped_nodes |= {x} 
                self.remove_edge(x,tn.idn,False)
                continue

            self.remove_edge(tn.idn,x,True)
            c.append(TNode(x,rdistance=tn.rdistance+1)) 

        self.untouched_nodes -= neighbors 
        tn.add_children(c)
        self.add_skipped_edges(tn.idn,skipped_nodes)
        return True  

    def remove_edge(self,e0,e1,is_directed:bool=False): 
        self.d_[e0] = self.d_[e0] - {e1} 
        if not is_directed: 
            self.d_[e1] = self.d_[e1] - {e0} 

    #------------------- skipped edges portion of graph 
    #------------------- component decomposition. 2nd half of 
    #------------------- tree decomposition for component. 

        #----------- add portion 
    def add_skipped_edges(self,p,skipped_nodes): 
        # order the skipped edges based on node-seniority 
        l = self.order_skipped_nodes(skipped_nodes)
        l = [(l_,p) for l_ in l] 
        self.skipped_edges = l + self.skipped_edges

    def order_skipped_nodes(self,skipped_nodes): 
        
        def sort_element(idn,neighbors): 
            if type(self.prg) != type(None): 
                neighbors = prg_seqsort(neighbors,self.prg) 

            if idn not in l: 
                l.append(idn) 
            index = l.index(idn) 

            while len(neighbors) > 0: 
                q = neighbors.pop(0) 
                if q in l: 
                    i = l.index(q) 
                    q = l.pop(i)
                    if i < index: 
                        index -= 1 
                    l.insert(index+1,q) 
                    continue  
                l.insert(index+1,q) 

        l = []
        done = set()
        skipped_nodes = list(skipped_nodes)
        if type(self.prg) != type(None): 
            skipped_nodes = prg_seqsort(skipped_nodes,self.prg)

        while len(skipped_nodes) > 0: 
            c = skipped_nodes.pop(0) 
            neighbors = self.d_[c] - done 
            sort_element(c,list(neighbors))
            done |= {c}
        return l 

        #----------- clear portion 

    def tree_from_skipped_edge(self): 
        stat = self.init_skipped_edge_to_tree() 
        if not stat: 
            return False 

        while len(self.dfs_nodecache) > 0: 
            tn = self.dfs_nodecache.pop(0)
            self.next_from_TNode__skipped_edges(tn) 
        return True 

    def init_skipped_edge_to_tree(self): 
        if len(self.skipped_edges) == 0: 
            return False 
        
        q = self.skipped_edges[0] 
        tn = TNode(q[0],root_stat=True,rdistance=0) 
        cx = self.children_in_skipped_edges(tn) 
        self.add_children_to_tn(tn,cx)
        self.decomp.append(tn) 
        self.dfs_nodecache.append(tn)
        tn.scached = True 
        return True

    def next_from_TNode__skipped_edges(self,tn): 
        q = next(tn) 
        if type(q) == type(None): return False 
        if not q.scached: 
            cx = self.children_in_skipped_edges(q)
            self.add_children_to_tn(q,cx) 
            q.scached = True 
        self.dfs_nodecache.insert(0,tn)
        self.dfs_nodecache.insert(0,q) 

    def children_in_skipped_edges(self,tn): 
        i = 0 
        cx = [] 
        while i < len(self.skipped_edges): 
            q = self.skipped_edges[i]
            if q[0] == tn.idn: 
                cx.append(q[1]) 
                self.skipped_edges.pop(i) 
            else: 
                i += 1 
        return cx 

    def add_children_to_tn(self,tn,cx_idn):
        cxnode = [TNode(cx_,rdistance=tn.rdistance+1) for cx_ in cx_idn]
        tn.add_children(cxnode)
