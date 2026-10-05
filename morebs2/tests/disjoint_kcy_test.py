from morebs2.disjoint_kcy import * 
import unittest

'''
py -m morebs2.tests.disjoint_kcy_test  
'''
class DisjointKCycleIteratorClass(unittest.TestCase):

    def test__DisjointKCyclesIterator__case_1(self):
        cperiod_seq = [4,5,7,13,2] 

        Q = DisjointKCyclesIterator(cperiod_seq,hop=1) 

        L = [] 
        for _ in range(10): 
            L.append(next(Q)) 

        L = np.array(L) 

        A = np.array([[0., 0., 0., 0., 0.], [1., 1., 1., 1., 1.],\
            [2., 2., 2., 2., 0.], [3., 3., 3., 3., 1.],\
            [0., 4., 4., 4., 0.], [1., 0., 5., 5., 1.],\
            [2., 1., 6., 6., 0.], [3., 2., 0., 7., 1.],\
            [0., 3., 1., 8., 0.], [1., 4., 2., 9., 1.]])

        assert np.all(A==L) 

        # test method<reverse> 
        assert np.all(Q.reverse() == A[-2]) 
        assert np.all(Q.reverse() == A[-3]) 

        assert np.all(next(Q) == A[-4]) 
        assert np.all(next(Q) == A[-3]) 

        for _ in range(2): next(Q) 

        # test method<update_hop> 
        Q.update_hop(3,2)  

        assert np.all(next(Q) == np.array([ 2.,  0.,  3., 10.,  0.]))
        assert np.all(next(Q) == np.array([ 3.,  1.,  6., 11.,  1.]))

if __name__ == '__main__':
    unittest.main()