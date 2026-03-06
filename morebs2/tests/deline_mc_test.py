from morebs2.deline_mc import * 
#from collections import Counter 
#import numpy as np
import unittest
import time 

def DLineateMCApproximators_sample_dataset(): 

    prg = prg__LCG(-1615.4,-3454.66,-919.16,-9191.4545) 
    def prg0(): 
        return modulo_in_range(prg(),[-10.,10]) 

    def prg1(): 
        return modulo_in_range(prg(),[5,15]) 

    def prg2(): 
        return modulo_in_range(prg(),[12,32])  

    pv0 = prg__single_to_nvec(prg0,2) 
    pv1 = prg__single_to_nvec(prg1,2) 
    pv2 = prg__single_to_nvec(prg2,2) 

    num_samples = 2000 
    xy0 = [pv0() for _ in range(num_samples)] 
    xy1 = [pv1() for _ in range(num_samples)] 
    xy2 = [pv2() for _ in range(num_samples)] 
    xy = np.array(xy0 + xy1 + xy2)
    L = [0 for _ in range(num_samples)] 
    L1 = [1 for _ in range(num_samples)] 
    L2 = [2 for _ in range(num_samples)] 
    L = np.array(L + L1 + L2).reshape((num_samples *3,1))
    xyl = np.hstack((xy,L))
    return xyl 

'''
py -m morebs2.tests.deline_mc_test  
'''
class DLineateMCApproximatorsClass(unittest.TestCase):

    """
    total runtime on personal device: ~ 205 seconds. 
    runtime to find classifier: ~ 40 seconds.
    runtime to classify: ~ (205 - 40) 
    number of samples: 6K 
    """
    def test__DLineateMCApproximators__one_classification__case_1(self):
        xyl = DLineateMCApproximators_sample_dataset() 
        d1 = DLineateMCApproximators(xyl,20,dmethod="nodup",prg=prg__LCG(-1615.4,-3454.66,-919.16,-9191.4545))

        t = time.time() 
        d1.full_run() 
        t2  = time.time() 
        print("search runtime: ",t2 - t)
        c = 0 
        for (i,x) in enumerate(xyl): 
            x2 = x[:2]
            rx = d1.one_classification(x2)
            #print("i: ",i, " label: ",x[2], " got: ",rx)
            c += int(x[2] == rx) 

        print("classification runtime: ",time.time() - t2)


if __name__ == "__main__":
    unittest.main()
