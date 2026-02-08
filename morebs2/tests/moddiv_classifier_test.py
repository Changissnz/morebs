from morebs2.numerical_space_data_generator import * 
from morebs2.modular_labeller import * 
from morebs2.moddiv_classifier import * 
import unittest 

# similar to the one used to test <RecursiveOneDimClassifier> 
def ModDivClassifier_sample_dataset_1(): 
    bounds = np.zeros((5,2)) 
    bounds[:,0] -= 15 
    bounds[:,1] += 15 

    startPoint = np.zeros((5,)) - 15 
    columnOrder = np.array([0,1,2,4,3]) 
    ssi_hop = 3 

    #additionalUpdateArgs = 
    rch = sample_rch_1_with_update(deepcopy(bounds),deepcopy(bounds),ssi_hop,0.1)

    rch2 = sample_rch_1_with_update(deepcopy(bounds),deepcopy(bounds),ssi_hop,0.1)
    cv = 0.4
    rssi2 = ResplattingSearchSpaceIterator(bounds, startPoint, \
        columnOrder, SSIHop = ssi_hop,resplattingMode = ("relevance zoom",rch), additionalUpdateArgs = (cv,))

    D = np.array([next(rssi2) for _ in range(1000)]) 
    L = np.array([label_vector__type_uniform_partition_index(v,[-15,15],2,num_labels=3) for v in D])
    return D,L 

class TestModDivClassifierClass(unittest.TestCase):

    '''
    '''
    def test__ModDivClassifier__make__case_1(self):
        D,L = ModDivClassifier_sample_dataset_1() 

        prg = prg__LCG(46.5,-13.44,788.9,988.7) 
        mdc = ModDivClassifier(D,L,prg,total_indexset_candidates=10,attempts_per=50,num_solutions=4)
        mdc.make() 
        msol = [s[1] for s in mdc.solutions]
        assert msol == [1000, 694, 586, 561]

        ###

        mdc2 = ModDivClassifier(D,L,prg,total_indexset_candidates=float('inf'),attempts_per=50,num_solutions=5)
        mdc2.make() 
        msol2 = [s[1] for s in mdc2.solutions]
        assert msol2 == [1000, 757, 703, 694, 667]


if __name__ == "__main__":
    unittest.main()