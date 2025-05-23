from morebs2 import g2tdecomp
import unittest

'''
python -m morebs2.tests.g2tdecomp_test  
'''
class G2TDecompClass(unittest.TestCase):

    def test__G2TDecomp__decomp_case1(self):
        D = g2tdecomp.defaultdict(set) 
        D[0] = set([1])
        D[1] = set([2])
        D[2] = set([3])
        D[3] = set([0])

        """
        rnodes = [0]
        gd = g2tdecomp.G2TDecomp(D,1,decomp_rootnodes=rnodes,prg=None)
        gd.decompose()

        assert len(gd.decompositions) == 1 
        assert len(gd.decompositions[0]) == 2
        q = gd.decompositions[0][0]
        d,rd = g2tdecomp.TNode.dfs(q,False,True,True) 
        assert d == g2tdecomp.defaultdict(set,{0: {1}, 1: {2}, 2: {3}, 3: set()})
        assert rd == 3

        q = gd.decompositions[0][1]
        d,rd = g2tdecomp.TNode.dfs(q,False,True,True) 
        assert d == g2tdecomp.defaultdict(set, {3: {0}, 0: set()})
        assert rd == 1
        return 
        """ 


if __name__ == '__main__':
    unittest.main()