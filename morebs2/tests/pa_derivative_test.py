from morebs2.pa_derivative import * 
import unittest

'''
py -m morebs2.tests.pa_derivative_test  
'''
class TestVectorPiecewiseAdditiveDerivativeClass(unittest.TestCase):

    '''
    '''
    def test__VectorPiecewiseAdditiveDerivative__next__case_1(self):

        length = 4 
        sum_info = 25 
        prg = prg__LCG(45.5,-155.6,115.3,1000.679) 
        prg = prg__LCG(22.1,34.3,-18.0,1000.55)
        segment_size = 5 
        record_derivative_info = True  
        allow_sign_change = False 

        vd = VectorPiecewiseAdditiveDerivative(length,sum_info,prg,segment_size,record_derivative_info=True) 

        V = [] 
        for _ in range(segment_size): 
            x = next(vd) 
            V.append(x) 
            ##print(vd.derivative_info("sign-all"))

        S = np.zeros((length,))
        for v in V: 
            S += v 
        assert np.sum(S) == sum_info 

        sum_info2 = np.array([10,5,-17,-35])
        segment_size2 = 10 
        vd.reset(sum_info2,segment_size2)

        V2 = [] 
        for _ in range(segment_size2): 
            x = next(vd) 
            V2.append(x) 
            ##print(vd.derivative_info("sign-all"))

        S2 = np.zeros((length,))
        for v in vd.record: 
            S2 += v 
        assert equal_iterables(S2,sum_info2)


if __name__ == "__main__":
    unittest.main()
