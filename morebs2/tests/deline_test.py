from morebs2 import deline
from collections import Counter 
import numpy as np
import unittest

'''
py -m morebs2.tests.deline_test  
'''
class DelineClass(unittest.TestCase):

    def test__DLineate22__collect_break_points__case_1(self):
        print("DLineat22 test 1")
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

    def test__DLineate22__collect_break_points__case_2(self):

        print("DLineat22 test 2")

        # clockwise 
        data = deline.test_dataset__Dlineate22_2()
        dl = deline.DLineate22(data)
        dl.preprocess()
        dl.collect_break_points()

        l = np.array([[ 5.,  0.],\
            [ 5., 15.]])

        r = np.array([[20., 15.],\
            [20.,  0.]])

        t = np.array([[ 5., 15.],\
            [20., 15.]])

        b = np.array([[20.,  0.],\
            [ 5.,  0.]])

        assert np.all(dl.d.d['l'] == l),"got {}".format(dl.d.d['l'])
        assert np.all(dl.d.d['r'] == r),"got {}".format(dl.d.d['r'])
        assert np.all(dl.d.d['t'] == t),"got {}".format(dl.d.d['t'])
        assert np.all(dl.d.d['b'] == b),"got {}".format(dl.d.d['b'])

        # counter-clockwise 
        data = deline.test_dataset__Dlineate22_2()
        dl = deline.DLineate22(data,False)
        dl.preprocess()
        dl.collect_break_points()

        l = np.array([[ 5., 15.],\
            [ 5.,  0.]])

        r = np.array([[20.,  0.],\
            [20., 15.]])

        t = np.array([[20., 15.],\
            [ 5., 15.]])

        b = np.array([[ 5.,  0.],\
            [20.,  0.]])

        assert np.all(dl.d.d['l'] == l),"got {}".format(dl.d.d['l'])
        assert np.all(dl.d.d['r'] == r),"got {}".format(dl.d.d['r'])
        assert np.all(dl.d.d['t'] == t),"got {}".format(dl.d.d['t'])
        assert np.all(dl.d.d['b'] == b),"got {}".format(dl.d.d['b'])

    def test__DLineate22__classify_points__case_3(self):
        print("DLineat22 test 3")

        td3 = deline.test_dataset__Dlineate22_3()
        p = [12.5,7.5]

        dl = deline.DLineate22(np.copy(td3),dmethod="nocross")
        dl.preprocess()
        dl.collect_break_points()

        c = dl.d.classify_point(p)    
        assert c == 0.0, "dmethod=nocross incorrectly classifies"

        dl = deline.DLineate22(np.copy(td3),dmethod="nodup")
        dl.preprocess()
        dl.collect_break_points()

        c = dl.d.classify_point(p)    
        assert c == 0.0, "dmethod=nodup incorrectly classifies"

        dl = deline.DLineate22(np.copy(td3),dmethod="nojag")
        dl.preprocess()
        dl.collect_break_points()

        c = dl.d.classify_point(p)    
        assert c == 0.0, "dmethod=nojag incorrectly classifies"

    def test__DLineate22__classify_points__case_4(self):

        print("DLineat22 test 4")

        td4 = deline.test_dataset__Dlineate22_4()
        dl = deline.DLineate22(np.copy(td4),dmethod="nodup")
        dl.preprocess()
        dl.collect_break_points()

        p = [12.5,7.5]
        c = dl.d.classify_point(p)
        assert c == 0, "misclassification for case 4, dmethod=nodup"

        dl = deline.DLineate22(np.copy(td4),dmethod="nocross")
        dl.preprocess()
        dl.collect_break_points()

        p = [12.5,7.5]
        c = dl.d.classify_point(p)
        assert c == 0, "misclassification for case 4, dmethod=nocross"

        dl = deline.DLineate22(np.copy(td4),dmethod="nojag")
        dl.preprocess()
        dl.collect_break_points()

        p = [12.5,7.5]
        c = dl.d.classify_point(p)
        assert c == 0, "misclassification for case 4, dmethod=nojag"

        p = [25,7.5]
        c = dl.d.classify_point(p)
        assert c == -1, "misclassification for case 4, dmethod=nojag"

    ## runtime is long 
    def test__DLineate22__classify_points__case_4(self):
        print("DLineat22 test 5")

        data = deline.test_dataset__DLineateMC_1()

        dl = deline.DLineate22(data,dmethod="nojag")
        dl.preprocess()
        dl.collect_break_points()
        
        c = 0 
        for x in data: 
            l = dl.d.classify_point(x[:2])
            l_ = x[2]
            if l == l_: c += 1 
            ##print("classifying ", l,l_)
        assert c == 1779, "got {}".format(c)

    def test__DLineate22__collect_break_points__AND__classify_point__case_1(self):

        print("DLineat22 test 6")

        data = deline.test_dataset__Dlineate22_1_v2()
        dl = deline.DLineate22(data)
        dl.preprocess()
        dl.collect_break_points()

        for q in dl.lpoints:
            c = dl.d.classify_point(q)
            assert c == 0
     
    def test__DLineate22__optimize_delineation__case_1(self):
        print("DLineat22 test 7")

        data = deline.test_dataset__Dlineate22_1_v3(500,[0.000001,1.])
        dl = deline.DLineate22(data)
        dl.preprocess()
        dl.collect_break_points()
        s = dl.optimize_delineation()
        assert s > 0, "optimization should reduce classification error!"
    
    def test__DLineate22__full_process__case_1(self):
        print("DLineat22 test 8")

        xyl = np.array([[5.,0.,0],\
                [7.5,15.,0],\
                [20.,-5.,0],\
                [1,15,1],\
                [3,4,2],\
                [17,15,2]])

        dl = deline.DLineate22(xyl,dmethod="nocross")
        q = dl.full_process()
        X = xyl[:,:2]
        L = xyl[:,2]
        results = [] 
        for d,l in zip(X,L): 
            l_ = dl.d.classify_point(d)
            #print("got {} actual {}".format(l_,l))
            results.append(l_) 

        assert results == [-1, -1, -1, 1, -1, -1]

    def test__DLineate22__full_process__case_2(self): 

        xyl = np.array([[0.,0.,0],\
                [10,0,0],\
                [0,10,0],\
                [10,10,0],\
                [5,5,1],\
                [15,15,2]])

        dl = deline.DLineate22(xyl,dmethod="nocross",target_min_label=False)
        q = dl.full_process()
        assert q[:3] == ([0, 1, 2, 3, 4], Counter({0: 4, 1: 1}), 4)

        X = xyl[:,:2]
        L = xyl[:,2]

        results = [] 
        for d,l in zip(X,L): 
            l_ = dl.d.classify_point(d)
            print("got {} actual {}".format(l_,l))
            results.append(l_) 

        assert results == [0,0,0,0,0,-1]

if __name__ == '__main__':
    unittest.main()