from morebs2.hmm_fb import * 
import unittest

'''
py -m morebs2.tests.hmm_fb_test  
'''
class TestForwardBackwardClass(unittest.TestCase):

    # NOTE: example from Wikipedia page @ 
    # https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    def test__ForwardBackward__run__case_1(self):

        T = {"r": {"r":0.7,"nr":0.3},"nr":{"r":0.3,"nr":0.7}} 
        B = {"u": {"r":0.9,"nr":0.2},"nu":{"r":0.1,"nr":0.8}} 

        S = ["u","u","nu","u","u"]
        fb = ForwardBackward(T,B)
        fb.run(S)

        X = [np.array([0.35306, 0.64694]), \
            np.array([0.13266, 0.86734]), \
            np.array([0.17957, 0.82043]), \
            np.array([0.69252, 0.30748]), \
            np.array([0.17959, 0.82041]),\
            np.array([0.13266, 0.86734])] 

        assert equal_iterables(fb.pr_smoothed,X) 

    # NOTE: results cannot be verified to be correct. 
    def test__ForwardBackward__run__case_2(self):

        T2 = {"s": {"s": 0.7,"c":0.2,"r":0.1},\
            "c": {"s": 0.3,"c":0.5,"r":0.2},\
            "r": {"s": 0.1,"c":0.3,"r":0.6}}

        B2 = {"d": {"s": 0.8,"c":0.4,"r":0.1},\
            "lr": {"s": 0.15,"c":0.4,"r":0.3},\
            "hr": {"s": 0.04,"c":0.15,"r":0.4},\
            "st": {"s": 0.01,"c":0.05,"r":0.2}}

        S2 = ["d","lr","hr","st"]
        fb2 = ForwardBackward(T2,B2) 

        fb2.run(S2,np.array([0.35,0.25,0.4]))

        X2 = [np.array([0.32722, 0.13553, 0.53725]),\
            np.array([0.39289, 0.08627, 0.52084]),\
            np.array([0.53637, 0.32939, 0.13424]),\
            np.array([0.23099, 0.74413, 0.02488]),\
            np.array([0.13207, 0.85279, 0.01514])]

        assert equal_iterables(fb2.pr_smoothed,X2) 

if __name__ == '__main__':
    unittest.main()
