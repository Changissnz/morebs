from morebs2.numerical_generator import * 
from morebs2.contour_op import * 
import unittest 

def contour_points_sample_E(): 
    c = (40,50) 
    rr = (5,10) 
    psize = 10 
    prg = prg__LCG(235.52,-5253,352352,1435.33)

    return generate_contour_points(c,rr,psize,prg,rounding_depth=5)

'''
py -m morebs2.tests.contour_op_test   
'''
class ContourInterpolationClass(unittest.TestCase):

    def test__ContourInterpolation__generate_points__case_1(self): 
        C = contour_points_sample_E() 
        assert C.shape == (11,3)

        CI = ContourInterpolation(C) 
        px = CI.generate_points(5)
        assert px.shape == (77,2) 

    def test__ContourInterpolation__generate_points_bt_cpoint_pair__case_1(self): 
        C = contour_points_sample_E() 
        CI = ContourInterpolation(C,add_noise=False) 

        # subcase 1: no noise 
        start_index = 0 
        end_index = 1

        p0,p1 = CI.cpoints[start_index],CI.cpoints[end_index]
        xs = sorted([p0[1],p1[1]])
        ys = sorted([p0[2],p1[2]])

        q = CI.generate_points_bt_cpoint_pair(start_index,5)

        for q_ in q: 
            assert xs[0] <= q_[0] <= xs[1] 
            assert ys[0] <= q_[1] <= ys[1] 

        # subcase 2: add noise 
        CI.prg = prg__LCG(235.52,-5253,352352,1435.33)
        CI.add_noise = True 
        start_index = 6 
        end_index = 7

        p0,p1 = CI.cpoints[start_index],CI.cpoints[end_index]
        xs = sorted([p0[1],p1[1]])
        ys = sorted([p0[2],p1[2]])

        q = CI.generate_points_bt_cpoint_pair(start_index,10)
        p = 0 
        for q_ in q:
            if xs[0] <= q_[0] <= xs[1] and ys[0] <= q_[1] <= ys[1]: 
                p += 1
        assert p == 11 

if __name__ == "__main__":
    unittest.main()