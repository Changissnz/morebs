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

        assert equal_iterables(fb.pr_smoothed,X), "got {}".format(fb.pr_smoothed)

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

        X2 = [np.array([0.35938, 0.17825, 0.46238]),\
        np.array([0.37151, 0.10754, 0.52095]),\
        np.array([0.44296, 0.3891 , 0.16793]),\
        np.array([0.21099, 0.75499, 0.03401]),\
        np.array([0.17003, 0.80868, 0.02129])]

        assert equal_iterables(fb2.pr_smoothed,X2) 

if __name__ == '__main__':
    unittest.main()
