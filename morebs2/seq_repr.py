from .matrix_methods import * 

"""
given: a sequence S, either an np.array or a list 
output: sequence S' corresponding to `S`. Every 
    element of S' is a pair of the form 
    [0] value v of S at the j'th index 
    [1] number of contiguous indices, from the 
        j'th index in increasing order, with 
        values that equal v. 

EX: 
S = <0,0,1,2,1,2,3,3,4,4,4>
S' = <(0,2),(1,1),(2,1),(1,1),(2,1),(3,2),(4,2)> 
"""
def contiguous_repr__sequence(S):
    assert is_vector(S) or type(S) == list
    assert len(S) > 0

    q = [[S[0],1]]
    ref = q[-1]

    for i in range(1,len(S)):
        if S[i] == ref[0]:
            ref[1] += 1
        else:
            ref_ = [S[i],1]
            q.append(ref_) 
            ref = ref_
    return q 