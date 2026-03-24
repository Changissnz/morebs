from morebs2.sr_struct import * 
import unittest

#########################################################

'''
py -m morebs2.tests.sr_struct_test   
'''
class TestSimulatedRecursionNode(unittest.TestCase):

    def test__SimulatedRecursionNode__next__case_1(self):#

        def pre_delta_function(x0,x1,x2): 
            x0 += 1 
            x1 -= 1 
            x2 += 1 
            return [x0,x1,x2]

        c0 = lambda x0,x1,x2: x0 + x1 + x2 > 100 
        c1 = None 
        c2 = lambda x0,x1,x2: x0 + x1 + x2 
        c3 = lambda x0,x1,x2: (x0 + x1 + x2) % 131 == 0

        C = (c0,c1,c2,c3)

        c4 = lambda x0,x1,x2: (x0 + x1 + x2) % 78 == 0
        C2 = (None,None,c2,c4)

        S = SimulatedRecursionNode(pre_delta_function,[C],C2) 
        p = [23,-5,16] 

        S.load_parameters(p) 

        while not S.terminated: 
            next(S) 
        assert sum(S.parameters) == 78 
        assert S.parameters == [67, -49, 60]
        assert len(S.output_seq) == 44, "got {}".format(len(S.output_seq))
        assert S.output_seq[-1] == 78 

if __name__ == '__main__':
    unittest.main()
