from .graph_basics import * 

"""
Auxiliary method for <calculate_dependency_order_for_IsoRing_list>.
"""
def order_element_in_dependency_order(depANDcodep_map,ooc,index):

    cds = ooc[index]
    # get all dependencies 
    dep = set() 
    for idn in cds:  
        dep |= depANDcodep_map[idn][0] 

    # case: codependencies cannot be dependencies 
    inter = dep.intersection(ooc[index]) 
    if len(inter) > 0: 
        return None,False  

    #print("DEP FOR {}: ".format(ooc[index],dep))  
    qi = index 
    x = None  
    for i in range(index + 1,len(ooc)): 
        inter = dep.intersection(ooc[i])  
        #print("OOC: ",ooc[i]) 
        if len(inter) > 0: 
            #print("YES") 
            x = i 
    
    if type(x) == type(None): 
        return -1,True 

    # pop element 
    q = ooc.pop(index) 
    ooc.insert(x,q) 
    return x,True   

def calculate_codependency_order(depANDcodep_map): 
    ooc = [] 
    num_no_deps = 0 
    # start with codependencies and verify
    elements = sorted(depANDcodep_map.keys()) 
    for elem in elements:
        
        ds = depANDcodep_map[elem][0] 
        if len(ds) == 0: 
            num_no_deps += 1 

        cds = depANDcodep_map[elem][1]
        cds = cds | {elem}  
        qi = index_of_element_in_setseq(ooc,elem)  

        if type(qi) == type(None): 
            ooc.append(cds) 
            continue 

        if ooc[qi] != cds: 
            return None,False,num_no_deps

    return ooc,True,num_no_deps

def calculate_dependency_order_(ooc,depANDcodep_map):  
    qx = deepcopy(ooc) 
    for i in range(len(ooc)):
        rx = qx[i]
        rx_ = next(iter(rx)) 
        j = index_of_element_in_setseq(ooc,rx_)
        index,stat = order_element_in_dependency_order(depANDcodep_map,ooc,j) 

        # case: error, co-dep cannot be dep 
        if not stat: 
            return None,False 
    return ooc,True 

"""
Calculates a dependency ordering using depANDcodep_map,
    element idn -> (dependency set, codependency set). 
During this calculation, determines if dependencies and 
codependencies of every <IsoRing> result in consistentn 
<IsoRingedChain>. 
"""
def calculate_dependency_order(depANDcodep_map):  

    ooc,stat,num_no_deps = calculate_codependency_order(depANDcodep_map) 
    if not stat: 
        return ooc,stat 

    # order elements in first scan
    for _ in range(num_no_deps): 
        ooc,stat = calculate_dependency_order_(ooc,depANDcodep_map)
        if not stat: 
            return ooc,stat 

    # check for contradictions in second scan
    for i in range(len(ooc)): 
        index, stat = order_element_in_dependency_order(depANDcodep_map,ooc,i)
        if index != -1: 
            return None,False 

    return ooc,True 

class DependencyGraph:

    def __init__(self,depANDcodep_map): 
        self.depANDcodep_map = depANDcodep_map
        self.setseq = None
        self.G = None 
        self.heads = None  
        self.ooc,self.stat,self.num_no_deps = None,None,None 
        self.active_proc_finstat = False 
        return 

    def process_order(self,require_proper_order:bool=True): 
        self.active_proc_finstat = True  
        if require_proper_order: 
            self.ooc,self.stat = calculate_dependency_order(self.depANDcodep_map)
            self.active_proc_finstat = False  
            return 

        self.ooc,self.stat,self.num_no_deps = calculate_codependency_order(self.depANDcodep_map)
        return

    def next_order(self): 
        if not self.active_proc_finstat: return False 

        ooc,self.stat = calculate_dependency_order_(deepcopy(self.ooc),self.depANDcodep_map)
        if not self.stat: 
            self.active_proc_finstat = False 

        if ooc == self.ooc: 
            self.active_proc_finstat = False 
        elif type(ooc) == type(None): 
            self.active_proc_finstat = False 
        else: 
            self.ooc = ooc 

    def to_graph(self): 
        assert type(self.ooc) != type(None) 

        self.G = defaultdict(set) 
        for i in range(len(self.ooc)): 
            self.next_nodeset(i)
        return self.G 

    def next_nodeset(self,index): 
        if index == 0: self.heads = deepcopy(self.ooc[index]) 

        for x in self.ooc[index]: 
            self.G[x] = self.ooc[index] - {x} 

        if index == 0: 
            return 

        q = self.ooc[index-1] 
        for x in self.ooc[index]: 
            for x2 in q: 
                self.G[x2] |= {x} 
        return