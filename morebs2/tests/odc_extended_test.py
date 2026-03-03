from .classification_test_cases import * 
from morebs2.modular_labeller import * 
from morebs2.odc_extended import * 
import unittest

'''
py -m morebs2.tests.odc_extended_test  
'''
class AdditiveAdjustedRecursiveODCClass(unittest.TestCase):

    def test__AdditiveAdjustedRecursiveODC__score_accuracy_case1(self):

        D = OneDimClassifier_test_dataset_1() 
        L = np.array([label_vector__type_binary_alt_sum(v) for v in D]) 

        # subcase 1
        prg = prg__LCG(-56.7,122.3,-31.6,-3121.66) 
        rodc = AdditiveAdjustedRecursiveODC(D,L,prg,0)
        rodc.fit() 

        c = rodc.score_accuracy(D,L)
        c2 = rodc.score_accuracy(D,L)
        assert c == c2 == 1000, "got {}".format(c)

        # subcase 2 
        rodc = AdditiveAdjustedRecursiveODC(D,L,prg,1)
        rodc.fit() 

        c = rodc.score_accuracy(D,L)
        c2 = rodc.score_accuracy(D,L)
        assert c == c2 == 904, "got {}".format(c)


        return

if __name__ == '__main__':
    unittest.main()