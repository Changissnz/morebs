'''
methods on integers such as multiple-finding
'''
from .poly_struct import * 

def all_multiples(i):
    l = abs(int(i / 2))
    m = set()
    for j in range(1,l + 1):
        if j in m: break
        if not i % j:
            m |= set([j,int(i/j)])
    return m

def all_multiple_pairs(i):
    if i == 1: return [(1,1)]
    if i == -1: return [(-1,1),(1,-1)]
    
    l = abs(int(i / 2))
    m = []
    s = set()
    for j in range(1,l + 1):
        if not i % j:
            if j in s: break
            if int(i/j) in s: break
            m.append((j,int(i/j)))
            m.append((int(i/j),j))
            s |= set([j,int(i/j)])

            # case: negative
            
            if i < 0:
                s |= set([-j,-int(i/j)])
                m.append((-j,-int(i/j)))
                m.append((-int(i/j),-j))            
    return m


def is_rational(f):
    assert f >= -1. and f <= 1., "invalid float"
    return "TODO"

