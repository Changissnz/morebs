from morebs2 import deline
import numpy as np

import unittest

class DelineClass(unittest.TestCase):

    def test__DLineate22__collect_break_points__case_1(self):
        data = deline.test_dataset__Dlineate22_1()
        dl = deline.DLineate22(data)
        dl.preprocess()
        dl.collect_break_points()

        l = np.array([[ 5.,  0.],\
        [5.,6.],\
        [ 5.,  9.],\
        [ 5., 12.],\
        [ 5., 15.]])
        
        r = np.array([[20., 15.],\
        [15., 12.],\
        [15.,  9.],\
        [25, 7.5],\
        [20.,  0.]])

        t = np.array([[ 5. , 15.],\
        [15. , 12.],\
        [20. , 15.],\
        [25. ,  7.5]])

        b = np.array([[25. ,  7.5],\
        [20. ,  0.],\
        [15. ,  9.],\
        [ 5. ,  0.]])

        assert np.all(dl.d.d['l'] == l),"got {}".format(dl.d.d['l'])
        assert np.all(dl.d.d['r'] == r),"got {}".format(dl.d.d['r'])
        assert np.all(dl.d.d['t'] == t),"got {}".format(dl.d.d['t'])
        assert np.all(dl.d.d['b'] == b),"got {}".format(dl.d.d['b'])

    def test__DLineate22__collect_break_points__AND__classify_point__case_1(self):
        data = deline.test_dataset__Dlineate22_1_v2()
        dl = deline.DLineate22(data)
        dl.preprocess()
        dl.collect_break_points()

        for q in dl.lpoints:
            c = dl.d.classify_point(q)
            assert c == 0


if __name__ == '__main__':
    unittest.main()