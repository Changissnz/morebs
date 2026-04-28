from .matrix_methods import * 

"""
pr2label_vec := list, element is (pr,label) 

NOTE: pr does not have to be confined to [0.,1.]. Probability values in 
      `pr2label_vec` do not have to sum to 1. All probability values are 
      positive. 
"""
def probability_to_label(pr2label_vec,pr): 
    assert type(pr2label_vec) == list 
    assert is_number(pr) 

    s = 0 
    for x in pr2label_vec: 
        assert len(x) == 2 
        r = [s,s+x[0]] 
        if r[0] <= pr <= r[1]: 
            return x[1] 
        s = s + x[0] 
    return None 


