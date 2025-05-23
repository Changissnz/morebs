from morebs2 import g2tdecomp
import unittest

'''
python -m morebs2.tests.g2tdecomp_test  
'''
class G2TDecompClass(unittest.TestCase):

    def test__G2TDecomp__decomp_case1(self):
        D = defaultdict(set) 
        D[0] = set([1])
        D[1] = set([2])
        D[2] = set([3])
        D[3] = set([0])

        rnodes = [0]
        gd = G2TDecomp(D,1,decomp_rootnodes=rnodes,prg=None)
        gd.decompose()

        assert len(gd.decompositions) == 1 
        assert len(gd.decompositions[0]) == 2
        q = gd.decompositions[0][0]
        d = TNode.dfs(q,False,True,True) 
        assert d == defaultdict(set,{0: {1}, 1: {2}, 2: {3}, 3: set()})

        q = gd.decompositions[0][1]
        d = TNode.dfs(q,False,True,True) 
        assert d == defaultdict(set, {3: {0}, 0: set()})
        return 


if __name__ == '__main__':
    unittest.main()