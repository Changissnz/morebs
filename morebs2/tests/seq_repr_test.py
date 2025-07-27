from morebs2 import seq_repr
import numpy as np 
import unittest

#########################################################

'''
python -m morebs2.tests.seq_repr_test  
'''
class TestSeqReprMethods(unittest.TestCase):

    def test__contiguous_repr__sequence__case1(self):#
        S = np.array([1,1,1,1,3,2,3,2,3,3,4,4,4,4,5,6,7,7,7,8,8,10,11,12,13,14,15,15]) 
        q = seq_repr.contiguous_repr__sequence(S)

        sol = [[1,4],\
            [3,1],[2,1],\
                [3,1],[2,1],\
                [3,2],[4,4],\
                [5,1],[6,1],\
                [7,3],[8,2],\
                [10,1],[11,1],\
                [12,1],[13,1],\
                [14,1],[15,2]]

        assert q == sol 
        return
    

if __name__ == '__main__':
    unittest.main()
