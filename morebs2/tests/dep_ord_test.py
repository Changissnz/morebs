from morebs2.dep_ord import * 
#import numpy as np

import unittest

'''
py -m morebs2.tests.dep_ord_test  
'''
class DepOrderMethodsClass(unittest.TestCase):

    def test__calculate_dependency_order__case_1(self):

        dcd_map = {0: ({1,2},{4,5}),\
                1: ({3,6},{7,8}),\
                2: ({1},{9,10}),\
                3: ({6},set()),\
                4: (set(),{0,5}),\
                5: (set(),{0,4}),\
                6: (set(),set()),\
                7: (set(),{1,8}),\
                8: (set(),{1,7}),\
                9: (set(),{2,10}),\
                10: (set(),{2,9})}

        q,stat = calculate_dependency_order(dcd_map)
        assert q == [{6}, {3}, {8, 1, 7}, {9, 10, 2}, {0, 4, 5}]

    def test__calculate_dependency_order__case_2(self):

        dcd_map = {0: (set(), {12}), 1: ({0, 3, 4, 5, 6, 7, 8, 11, 12, 13}, set()), \
            2: ({0, 1, 3, 4, 5, 6, 7, 8, 11, 12, 13}, set()), 3: ({0, 4, 5, 6, 12}, {8, 11, 7}), \
            4: ({0, 12}, set()), 5: ({0, 12, 4}, set()), 6: ({0, 5, 12, 4}, set()), \
            7: ({0, 4, 5, 6, 12}, {8, 11, 3}), 8: ({0, 4, 5, 6, 12}, {11, 3, 7}), \
            9: ({0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13}, {14}), \
            10: ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14}, set()), \
            11: ({0, 4, 5, 6, 12}, {8, 3, 7}), 12: (set(), {0}), 13: ({0, 3, 4, 5, 6, 7, 8, 11, 12}, set()), \
            14: ({0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13}, {9})}

        q,stat = calculate_dependency_order(dcd_map)
        sol = [{0, 12}, {4}, {5}, {6}, {8, 3, 11, 7}, {13}, {1}, {2}, {9, 14}, {10}]
        assert q == sol 

    def test__calculate_dependency_order__case_3(self):

        dcd_map = {0: ({1,2},{4,5}),\
                1: ({2,3,6},{7,8}),\
                2: ({1},{9,10}),\
                3: ({6},set()),\
                4: (set(),{0,5}),\
                5: (set(),{0,4}),\
                6: (set(),set()),\
                7: (set(),{1,8}),\
                8: (set(),{1,7}),\
                9: (set(),{2,10}),\
                10: (set(),{2,9})}
        q,stat = calculate_dependency_order(dcd_map)
        assert not stat 

    def test__calculate_dependency_order__case_4(self):

        dcd_map = {0: (set(),{4,5}),\
                1: ({0},{7,8}),\
                2: ({1},{9,10}),\
                3: ({2},set()),\
                4: ({3},{0,5}),\
                5: (set(),{0,4}),\
                6: (set(),set()),\
                7: (set(),{1,8}),\
                8: (set(),{1,7}),\
                9: (set(),{2,10}),\
                10: (set(),{2,9})}
        q,stat = calculate_dependency_order(dcd_map)
        assert not stat 

    def test__calculate_dependency_order__case_5(self):

        dcd_map = {0: (set(),{1,2}),\
                1: (set(),{0,2}),\
                2: (set(),{0,1}),\
                3: ({1},{4}),\
                4: ({2},{3}),\
                5: ({4},set()),\
                6: ({1},{7}),\
                7: ({5},{6}),\
                8: (set(),{9,10,11}),\
                9: (set(),{8,10,11}),\
                10: (set(),{8,9,11}),\
                11: ({6},{8,9,10})}
        q,stat = calculate_dependency_order(dcd_map)
        assert q == [{0, 1, 2}, {3, 4}, {5}, {6, 7}, {8, 9, 10, 11}]

    def test__calculate_dependency_order__case_6(self):
        dcd_map = {12: (set(),{13,14}),\
                13: (set(),{12,14}),\
                14: (set(),{12,13}),\
                3: ({13},{4}),\
                4: ({14},{3}),\
                5: ({4},set()),\
                6: ({13},{7}),\
                7: ({5},{6}),\
                8: (set(),{9,10,11}),\
                9: (set(),{8,10,11}),\
                10: (set(),{8,9,11}),\
                11: ({6},{8,9,10})}
        q,stat = calculate_dependency_order(dcd_map)
        assert q == [{12, 13, 14}, {3, 4}, {5}, {6, 7}, {8, 9, 10, 11}]

    def test__calculate_dependency_order__case_7(self):

        dcd_map = {0: (set(),{1,2}),\
                1: ({2},set()),\
                2: (set(),set())} 
        q,stat = calculate_dependency_order(dcd_map)
        assert not stat 


if __name__ == '__main__':
    unittest.main()