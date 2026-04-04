from morebs2.frequency_generator import * 
from morebs2.numerical_generator import * 
import unittest

'''
py -m morebs2.tests.frequency_generator_test  
'''
class TestPoissonBasedFreqencyOutputterClass(unittest.TestCase):

    def test__PoissonBasedFreqencyOutputter__out__case_1(self):

        x = prg__LCG(112.33,-47.66,811.55,9106.75) 

        expected = 14 
        possible_range = [10,20] 

        # subcase 1
        P = PoissonBasedFreqencyOutputter(x)

        L = [] 
        for _ in range(10): 
            q = P.out(expected,possible_range)
            L.append(q) 
        assert L == [10, 13, 11, 14, 11, 10, 12, 17, 16, 17]

        # subcase 2
        x2 = prg__LCG(8917.33,-47.66,811.55,9105.75) 
        P2 = PoissonBasedFreqencyOutputter(x2)

        L2 = [] 
        for _ in range(10): 
            q = P2.out(expected,possible_range)
            L2.append(q) 
        assert L2 == [13, 11, 10, 16, 12, 18, 11, 12, 12, 12],"got {}".format(L2)


if __name__ == '__main__':
    unittest.main()