from morebs2 import seq_repr
import numpy as np 
import unittest

#########################################################

'''
py -m morebs2.tests.seq_repr_test  
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

    def test__MCSSearch__search__case1(self):
        L = [1,2,3,4,1,2,3,4,1,2,3,4,1,3,4,2,3,4] 
        ms = seq_repr.MCSSearch(L,cast_type=int,is_bfs=True)  
        ms.search() 

        r0 = ['3','4', '3,4']
        r1 = ['1', '2', '2,3', '2,3,4']
        r2 = ['1,2', '4,1', '1,2,3', '3,4,1', '1,2,3,4', '2,3,4,1', '1,2,3,4,1']
        r3 = ['4,1,2', '3,4,1,2', '4,1,2,3', '2,3,4,1,2', '3,4,1,2,3', \
            '4,1,2,3,4', '1,2,3,4,1,2', '2,3,4,1,2,3', '3,4,1,2,3,4', \
            '4,1,2,3,4,1', '1,2,3,4,1,2,3', '2,3,4,1,2,3,4', '3,4,1,2,3,4,1', \
            '1,2,3,4,1,2,3,4', '2,3,4,1,2,3,4,1', '1,2,3,4,1,2,3,4,1']
        R = [r0,r1,r2,r3] 

        for i in range(4):
            q = R[i] 
            q_ = ms.mcs_nth(i)
            assert q_ == q, "{} got {}".format(i,q_)

    def test__MCSSearch__default_kcomplexity__case1(self): 
        L = [1,2,3,4,1,2,3,4,1,2,3,4,1,3,4,2,3,4] 
        ms = seq_repr.MCSSearch(L,cast_type=int,is_bfs=True)  
        ms.search() 
        kq = ms.kcomplexity(diff_type="bool")

        kxdd = {4:86,3:16,2:7,1:4,0:3} 
        for j in range(0,5): 
            kc_ = ms.kcomplexity_at_nth_set(j,diff_type="bool")
            assert len(kc_) == kxdd[j], "{} got {}".format(j,len(kc_))

        qdd = ms.default_kcomplexity()
        assert qdd == 12.0, "got {}".format(qdd)

        qdd_ = ms.default_kcomplexity(diff_type2="best") 
        assert round(qdd_,5) == 11.33333, "got {}".format(qdd_) 

        qdd2 = ms.default_kcomplexity(diff_type="bool",basis="median")
        assert qdd2 == 5.0, "got {}".format(qdd2) 

        qdd3 = ms.default_kcomplexity(diff_type="bool",diff_type2="best",basis="median")
        assert qdd3 == 5.0, "got {}".format(qdd3) 

    def test__MCSSearch__default_kcomplexity__case2(self): 

        L = [1,2,3,4,1,2,3,4,1,2,3,4,1,3,4,2,3,4] 

        ms = seq_repr.MCSSearch(L,cast_type=int,is_bfs=True)  
        ms.search() 

        kcxx = ms.kcomplexity(keys=["1,2,3,4"],diff_type="bool",diff_type2="contiguous")
        kcxx2 = ms.kcomplexity(keys=["1,2,3,4"],diff_type="bool",diff_type2="best")
        assert kcxx == [('1,2,3,4', 5)]
        assert kcxx2 == [('1,2,3,4', 6)]

        L2 = [2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,3,4] 
        ms2 = seq_repr.MCSSearch(L2,cast_type=int,is_bfs=True)  
        ms2.search() 

        kcxx3 = ms2.kcomplexity(keys=["1,2,3,4"],diff_type="bool",diff_type2="contiguous")
        kcxx4 = ms2.kcomplexity(keys=["1,2,3,4"],diff_type="bool",diff_type2="best")
        assert kcxx3 == [('1,2,3,4', 2)]
        assert kcxx4 == [('1,2,3,4', 6)]

    def test__MCSSearch__cyclical_kernel_nth__AND__best_cyclical_kernel__case1(self): 
        # subcase 1 
        L = [1,3,2]
        L = L * 5
        ms = seq_repr.MCSSearch(L,cast_type=int,is_bfs=True)  
        ms.search() 

        M = ms.cyclical_kernel_nth(0)
        assert M == {'1': 10, '2': 10, '3': 10, '1,3': 9, '3,2': 9, '1,3,2': 0}

        K,s = ms.best_cyclical_kernel() 
        assert K == [1,3,2] and s == 0 

        # subcase 2 
        L = L[2:]
        ms = seq_repr.MCSSearch(L,cast_type=int,is_bfs=True)  
        ms.search() 

        M = ms.cyclical_kernel_nth(0)
        assert M == {'2': 8}

        K,s = ms.best_cyclical_kernel() 
        assert K == [2,1,3] and s == 0 

        # subcase 3 
        L = [1,2,3,4,5] * 5 
        L = L[3:] 
        ms = seq_repr.MCSSearch(L,cast_type=int,is_bfs=True)  
        ms.search() 

        M = ms.cyclical_kernel_nth(0)
        assert M == {'4': 17, '5': 17, '4,5': 16}

        K,s = ms.best_cyclical_kernel() 
        assert K == [4,5,1,2,3] and s == 0 

if __name__ == '__main__':
    unittest.main()
