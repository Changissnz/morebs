from morebs2 import distributions
import unittest

'''
py -m morebs2.tests.distributions_test  
'''
class TestDistributionsFile(unittest.TestCase):

    def test__poisson_distribution_pr__case_1(self):

        x0 = distributions.poisson_distribution_pr(10,10)
        x1 = distributions.poisson_distribution_pr(10,11)
        x2 = distributions.poisson_distribution_pr(10,9)
        x3 = distributions.poisson_distribution_pr(10,12)
        x4 = distributions.poisson_distribution_pr(10,8)

        x0 = round(x0,5)
        x1 = round(x1,5)
        x2 = round(x2,5)
        x3 = round(x3,5)
        x4 = round(x4,5)

        assert x0 == 0.12511
        assert x1 == 0.11374
        assert x2 == 0.12511
        assert x3 == 0.09478
        assert x4 == 0.1126


if __name__ == '__main__':
    unittest.main()