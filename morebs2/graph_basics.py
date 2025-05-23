from collections import defaultdict 
from copy import deepcopy 

def is_directed_graph(d):
    assert type(d) in {defaultdict,dict}
    
    for k,v in d.items(): 
        for v_ in v: 
            if v_ not in d: return False 
            if k not in d[v_]: return False 
    return True 