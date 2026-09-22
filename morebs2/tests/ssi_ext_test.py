from morebs2.ssi_ext import * 
import unittest

'''
py -m morebs2.tests.ssi_ext_test   
'''
class TestSSIExtClasses(unittest.TestCase):

    def test__MultiIndexIterator__case_1(self):
        q = [4,3,2,2] 

        R = MultiIndexIterator(q) 

        ans1 = [[0, 0, 0, 0],[1, 0, 0, 0],[2, 0, 0, 0],[3, 0, 0, 0],\
            [0, 1, 0, 0],[1, 1, 0, 0],[2, 1, 0, 0],[3, 1, 0, 0],\
            [0, 2, 0, 0],[1, 2, 0, 0],[2, 2, 0, 0],[3, 2, 0, 0],\
            [0, 0, 1, 0],[1, 0, 1, 0],[2, 0, 1, 0],[3, 0, 1, 0],\
            [0, 1, 1, 0],[1, 1, 1, 0],[2, 1, 1, 0],[3, 1, 1, 0],\
            [0, 2, 1, 0],[1, 2, 1, 0],[2, 2, 1, 0],[3, 2, 1, 0],\
            [0, 0, 0, 1],[1, 0, 0, 1],[2, 0, 0, 1],[3, 0, 0, 1],\
            [0, 1, 0, 1],[1, 1, 0, 1],[2, 1, 0, 1],[3, 1, 0, 1],\
            [0, 2, 0, 1],[1, 2, 0, 1],[2, 2, 0, 1],[3, 2, 0, 1],\
            [0, 0, 1, 1],[1, 0, 1, 1],[2, 0, 1, 1],[3, 0, 1, 1],\
            [0, 1, 1, 1],[1, 1, 1, 1],[2, 1, 1, 1],[3, 1, 1, 1],\
            [0, 2, 1, 1],[1, 2, 1, 1],[2, 2, 1, 1],[3, 2, 1, 1]] 

        for i in range(48): 
            x = next(R) 
            assert equal_iterables(ans1[i],x) 
        assert type(next(R)) == type(None)

    def test__OrderedSelection__case_1(self): 

        S = [(2,3),(5,6,7),(-1,1),(10,)] 
        R2 = OrderedSelection(S) 


        ans2 = [[2, 5, -1, 10],\
            [3, 5, -1, 10],\
            [2, 6, -1, 10],\
            [3, 6, -1, 10],\
            [2, 7, -1, 10],\
            [3, 7, -1, 10],\
            [2, 5, 1, 10],\
            [3, 5, 1, 10],\
            [2, 6, 1, 10],\
            [3, 6, 1, 10],\
            [2, 7, 1, 10],\
            [3, 7, 1, 10],\
            None,None,None] 

        Q2 = [] 
        for i in range(15): 
            q = next(R2)
            Q2.append(q) 

        assert ans2 == Q2 



if __name__ == '__main__':
    unittest.main()