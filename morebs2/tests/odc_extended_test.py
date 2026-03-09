from .classification_test_cases import * 
from morebs2.modular_labeller import * 
from morebs2.odc_extended import * 
import unittest

'''
py -m morebs2.tests.odc_extended_test  
'''
class AdditiveAdjustedRecursiveODCClass(unittest.TestCase):

    def test__AdditiveAdjustedRecursiveODC__score_accuracy_case_1(self):

        D = OneDimClassifier_test_dataset_1() 
        L = np.array([label_vector__type_binary_alt_sum(v) for v in D]) 

        # subcase 1
        prg = prg__LCG(-56.7,122.3,-31.6,-3121.66) 
        rodc = AdditiveAdjustedRecursiveODC(D,L,prg,0)
        rodc.fit() 

        c = rodc.score_accuracy(D,L)
        c2 = rodc.score_accuracy(D,L)
        c3 = rodc.score_accuracy(D,L)
        assert c == c2 == c3 == 1000, "got {},{}".format(c,c2)

        # subcase 2 
        rodc = AdditiveAdjustedRecursiveODC(D,L,prg,1)
        rodc.fit() 

        c = rodc.score_accuracy(D,L)
        c2 = rodc.score_accuracy(D,L)
        assert c == c2 == 904, "got {}".format(c)
        return

    """
    tests <ODCAdditiveAdjustmentNode> with half of its contiguous information 
    missing. 

    Diminished accuracy from 1000 to 559. 
    """ 
    def test__AdditiveAdjustedRecursiveODC__score_accuracy_case_2(self):
        D = OneDimClassifier_test_dataset_1() 
        L = np.array([label_vector__type_binary_alt_sum(v) for v in D]) 

        # subcase 1
        prg = prg__LCG(-56.7,122.3,-31.6,-3121.66) 
        rodc = AdditiveAdjustedRecursiveODC(D,L,prg,0)
        rodc.fit() 

        # try removing some of the indices
        node = rodc.root_ 
        cs = node.contiguous_sequence
        cs_ = [cs.pop(0),cs.pop(-1)]
        qx = prg_choose_n(cs,len(cs) // 2,prg__single_to_int(prg),is_unique_picker=True)
        qx = sorted(qx + cs_,key=lambda x: x[2]) 
        node.contiguous_sequence = qx 

        c = rodc.score_accuracy(D,L)
        assert c == 590,"got {}".format(c)

class MBAAARecursiveODCClass(unittest.TestCase):

    """
    tests demonstrate the varying accuracy of classifiers 
    based on different memory parameters, as well as 
    identifical parameters but different PRNG-based decisions. 
    """
    def test__MBAAARecursiveODC__score_accuracy_case_1(self):
        D = OneDimClassifier_test_dataset_1() 
        L2 = np.array([label_vector__type_binary_alt_sum(v) for v in D]) 
        
        ##q = deepcopy(D) 
        ##L2_ = deepcopy(L2)

        # subcase 1
        prg = prg__LCG(-56.7,122.3,-31.6,-3121.66) 

        rodc2 = MBAAARecursiveODC(D,L2,prg,0,max_mem_sz=2000,mem_ratio_per_node=0.95) 
        rodc2.fit() 
        c2 = rodc2.score_accuracy(D,L2)

        rodc3 = MBAAARecursiveODC(D,L2,prg,0,max_mem_sz=2000,mem_ratio_per_node=0.9) 
        rodc3.fit() 
        c3 = rodc3.score_accuracy(D,L2)

        rodc4 = MBAAARecursiveODC(D,L2,prg,0,max_mem_sz=2000,mem_ratio_per_node=0.75) 
        rodc4.fit() 
        c4 = rodc4.score_accuracy(D,L2)

        rodc5 = MBAAARecursiveODC(D,L2,prg,0,max_mem_sz=2000,mem_ratio_per_node=0.6) 
        rodc5.fit() 
        c5 = rodc5.score_accuracy(D,L2)

        rodc6 = MBAAARecursiveODC(D,L2,prg,0,max_mem_sz=2000,mem_ratio_per_node=0.5) 
        rodc6.fit() 
        c6 = rodc6.score_accuracy(D,L2)

        rodc7 = MBAAARecursiveODC(D,L2,prg,0,max_mem_sz=2000,mem_ratio_per_node=0.5) 
        rodc7.fit() 
        c7 = rodc7.score_accuracy(D,L2)

        rodc8 = MBAAARecursiveODC(D,L2,prg,0,max_mem_sz=2000,mem_ratio_per_node=0.95) 
        rodc8.fit() 
        c8 = rodc8.score_accuracy(D,L2)

        rodc9 = MBAAARecursiveODC(D,L2,prg,1,max_mem_sz=2000,mem_ratio_per_node=0.95) 
        rodc9.fit() 
        c9 = rodc9.score_accuracy(D,L2)

        CX = [c2,c3,c4,c5,c6,c7,c8,c9] 
        assert [987, 622, 473, 431, 501, 401, 675, 963]

if __name__ == '__main__':
    unittest.main()