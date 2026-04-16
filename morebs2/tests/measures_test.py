from morebs2.measures import *
import unittest

'''
py -m morebs2.tests.measures_test   
'''
class TestMeasuresFile(unittest.TestCase):

    def test_to_trinary_relation_v2__case_1(self):

        v1 = 10 
        v2 = np.array([3,40,-10,10,-20]) 

        r0 = to_trinary_relation_v2(v1,v2,zero_feature=False,abs_feature=True)
        assert np.all(r0 == np.array([ 1, -1,  0,  0, -1]))

        r1 = to_trinary_relation_v2(v1,v2,zero_feature=True,abs_feature=False) 
        assert np.all(r1 == np.array([ 1, -1,  1,  0,  1])) 

        v2_ = np.array([-3,-41,9,-11,20]) 
        r2 = to_trinary_relation_v2(v2,v2_,zero_feature=False,abs_feature=True) 
        assert np.all(r2 == np.array([ 0, -1,  1, -1,  0]))

        r3 = to_trinary_relation_v2(v2,v2_,zero_feature=True,abs_feature=False) 
        assert np.all(r3 == np.array([ 1,  1, -1,  1, -1]))
        
if __name__ == '__main__':
    unittest.main()
