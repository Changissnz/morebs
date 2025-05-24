from morebs2 import g2tdecomp
from .graph_basics_test_cases import * 
import unittest

'''
python -m morebs2.tests.g2tdecomp_test  
'''
class G2TDecompClass(unittest.TestCase):

    def test__G2TDecomp__decomp_case1(self):
        # soln 
        a01 = defaultdict(list, {0: [0, 2], 1: [0, 2], 3: [0, 0], 4: [0, 0], 2: [0, 2]})
        a0 = (6,a01)

        a11 = defaultdict(list, {1: [0, 2], 0: [0, 2], 3: [0, 0], 4: [0, 0], 2: [0, 0]})
        a1 = (4,a11)

        a21 = defaultdict(list, {2: [0, 2], 0: [0, 0], 1: [0, 0]})
        a2 = (2,a21) 

        a31 = defaultdict(list, {3: [0, 2], 0: [0, 0], 1: [0, 0]})
        a3 = (2,a31)

        a41 = defaultdict(list, {4: [0, 2], 0: [0, 0], 1: [0, 0]})
        a4 = (2,a41)

        sol0 = {0:a0,1:a1,2:a2,3:a3,4:a4} 

        D = graph_case_15() 
        gd = g2tdecomp.G2TDecomp(D,child_capacity=2)
        gd.decompose()

        ks = set()
        for x in gd.decompositions: 
            dx, md = g2tdecomp.TNode.dfs(x,display=False)
            q = sol0[x.idn] 
            ec = g2tdecomp.edge_count(dx)
            ncm = g2tdecomp.nc_degree_map(dx) 
            assert ec == q[0] 
            assert ncm == q[1]
            ks |= {x.idn}
        
        assert ks == set(sol0.keys())

    def test__G2TDecomp__decomp_case2(self):

        # sol'n 
        r = defaultdict(list, \
            {0: [0, 4], 1: [0, 2], 5: [0, 0], \
            6: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]})
        sol0 = (6,r)

        r7 = defaultdict(list, \
            {7: [0, 2], 12: [0, 2], 38: [0, 1], \
            11: [0, 1], 31: [0, 0], 41: [0, 0], \
            20: [0, 2], 15: [0, 0], 64: [0, 0]})

        sol7 = (8,r7)

        D = graph_case_11() 
        gd = g2tdecomp.G2TDecomp(D) 
        gd.decompose()

        for x in gd.decompositions: 
            dx, md = g2tdecomp.TNode.dfs(x,display=False)

            if x.idn == 0: 
                q1 = g2tdecomp.edge_count(dx) 
                q2 = g2tdecomp.nc_degree_map(dx) 
                assert q1 == sol0[0]
                assert q2 == sol0[1]

                sz0 = g2tdecomp.TNode.size_count(x)
                assert sz0 == 7 

            elif x.idn == 7: 
                q1 = g2tdecomp.edge_count(dx) 
                q2 = g2tdecomp.nc_degree_map(dx) 
                assert q1 == sol7[0]
                assert q2 == sol7[1]

                sz7 = g2tdecomp.TNode.size_count(x)
                assert sz7 == 9 

            elif x.idn == 11: 
                q1 = g2tdecomp.edge_count(dx) 
                q2 = g2tdecomp.nc_degree_map(dx) 
                assert q1 == 1

                sz11 = g2tdecomp.TNode.size_count(x)
                assert sz11 == 2 

        assert len(gd.decompositions) == 16, "actual {}".format(len(gd.decompositions))

if __name__ == '__main__':
    unittest.main()