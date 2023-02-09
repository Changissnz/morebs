from morebs2 import deline
import numpy as np

import unittest

class DelineClass(unittest.TestCase):

    def test__DLineate22__collect_break_points__case_1(self):
        data = deline.test_dataset__Dlineate22_1()
        dl = deline.DLineate22(data)
        dl.preprocess()
        dl.collect_break_points()

        l = np.array([[ 5.,  0.,  0.],\
        [ 5.,  9.,  0.],\
        [ 5., 12.,  0.],\
        [ 5., 15.,  0.]])
        
        r = np.array([[20., 15.,  0.],\
        [15., 12.,  0.],\
        [15.,  9.,  0.],\
        [20.,  0.,  0.]])

        t = np.array([[ 5. , 15. ,  0. ],\
        [15. , 12. ,  0. ],\
        [20. , 15. ,  0. ],\
        [25. ,  7.5,  0. ]])

        b = np.array([[25. ,  7.5,  0. ],\
        [20. ,  0. ,  0. ],\
        [15. ,  9. ,  0. ],\
        [ 5. ,  0. ,  0. ]])

        assert np.all(dl.d.d['l'] == l)
        assert np.all(dl.d.d['r'] == r)
        assert np.all(dl.d.d['t'] == t)
        assert np.all(dl.d.d['b'] == b)

if __name__ == '__main__':
    unittest.main()