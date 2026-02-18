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
            return None,False

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

    return ooc, True 
