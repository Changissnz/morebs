from morebs2.sr_struct import * 
import unittest

def SimulatedRecursionNode__sample_T(): 

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
    return S 


def SimulatedRecursionNode__sample_U(): 
    S = SimulatedRecursionNode__sample_T() 

    def pre_delta_function(x0,x1,x2): 
        x0 *= 2 
        x1 -= 117 
        x2 += 1919
        return [x0,x1,x2]

    S.set_pre_delta_function(pre_delta_function)

    return S 

#########################################################

'''
py -m morebs2.tests.sr_struct_test   
'''
class TestSimulatedRecursionNode(unittest.TestCase):

    def test__SimulatedRecursionNode__next__case_1(self):#

        S = SimulatedRecursionNode__sample_T()

        p = [23,-5,16] 
        S.load_parameters(p) 

        while not S.terminated: 
            next(S) 
        assert sum(S.parameters) == 78 
        assert S.parameters == [67, -49, 60]
        assert len(S.output_seq) == 44, "got {}".format(len(S.output_seq))
        assert S.output_seq[-1] == 78 

    def test__SimulatedRecursionNode__next__case_2(self):#

        S = SimulatedRecursionNode__sample_U()

        p = [23,-5,16] 
        S.load_parameters(p) 

        while not S.terminated: 
            next(S) 

        q = sum(S.parameters)
        assert q == 444884701618183536292024147
        assert q % 131 == 0 

if __name__ == '__main__':
    unittest.main()
