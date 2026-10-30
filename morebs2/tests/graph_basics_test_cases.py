from collections import defaultdict 

#------------- single component graphs of directed or undirected. 

def graph_case_1(): 
    D = defaultdict(set) 
    D[0] = set([1])
    D[1] = set([2,0])
    D[2] = set([0])
    D[3] = set([0])
    return D 

def graph_case_2():
    D = defaultdict(set) 
    D[0] = set([])
    D[1] = set([0])
    D[2] = set([0])
    D[3] = set([0])
    D[4] = set([0])
    return D 

def graph_case_3():
    D = defaultdict(set) 
    D[0] = set([1,2,3,4])
    D[1] = set([0])
    D[2] = set([0])
    D[3] = set([0])
    D[4] = set([0])
    return D 

def graph_case_4():
    D = defaultdict(set) 
    D[0] = set([1,2])
    D[1] = set([3,4])
    D[2] = set([5,6])
    D[3] = set([7])
    D[7] = set([0,3])
    return D 

def graph_case_5():
    D = defaultdict(set) 
    D[0] = set([1,2])
    D[1] = set([3,4])
    D[2] = set([5,6])
    D[3] = set([7])
    D[7] = set([0,3])
    return D 

def graph_case_6(): 
    D = defaultdict(set) 
    D[0] = set([1,2])
    D[1] = set([0,3,4])
    D[2] = set([0,5,6])
    D[3] = set([1,4])
    D[4] = set([1,3])
    D[5] = set([2])
    D[6] = set([2])
    return D 

def graph_case_7():
    D = defaultdict(set) 
    D[0] = set([1,2])
    D[1] = set([3,4])
    D[2] = set([5,6])
    D[3] = set([7])
    D[7] = set([0,3,8])
    D[8] = set([7,9,10])
    D[9] = set([8,10])
    return D 

def graph_case_8(): 
    D = defaultdict(set) 
    D[0] = set([1,2])
    D[1] = set([3,4])
    D[2] = set([5,6])
    D[3] = set([7])
    D[7] = set([0,3,8])
    D[8] = set([7,9,10])
    D[9] = set([8,10])
    D[10] = set([8,9])
    return D 

def graph_case_9(): 
    D = defaultdict(set) 
    D[0] = set([1,2,3,4]) 
    D[1] = set([0,5,6])
    D[2] = set([0])
    D[3] = set([0])
    D[4] = set([0])
    D[5] = set([0])
    return D 


def graph_case_12(): 
    D = defaultdict(set) 
    D[0] = set([1,2,3,4])
    D[1] = set([0,2,8])
    D[2] = set([0,1,6,7])
    D[3] = set([10])
    D[4] = set([10])
    D[5] = set([0])
    D[6] = set([7])
    D[7] = set([6])
    D[8] = set([1,3,9])
    D[9] = set([8,10])
    D[10] = set([3,4,9])
    return D 

def graph_case_13(): 
    D = defaultdict(set) 
    D[0] = set([6,7])
    D[6] = set([8,9])
    D[7] = set([0])
    D[8] = set([0,10])
    D[9] = set([6,8])
    D[10] = set([])
    return D 


def graph_case_14(): 
    D = graph_case_13() 
    D[10] |= {11,12}
    D[11] |= {13}
    D[12] |= {10,13}
    # delete below edge for different result 
    D[13] |= {10}
    return D 

def graph_case_15(): 
    D = defaultdict(set) 
    D[0] = set([1,2,3,4]) 
    D[1] = set([0,2,3,4]) 
    D[2] = set([1,0,3,4]) 
    D[3] = set([1,2,0,4]) 
    D[4] = set([1,2,3,0]) 
    return D 

#----------------------------------------
#----------- multi-component graphs 


def graph_case_10(): 
    D = defaultdict(set) 
    D[0] = set([1,2,3,4]) 
    D[1] = set([0,2,3,4])
    D[2] = set([0,1,3,4])
    D[3] = set([0,1,2,4])
    D[4] = set([1,2,3,0])

    D[7] = set([8,9]) 
    D[8] = set([7,9])
    D[9] = set([7,8])

    D[10] = set([111]) 
    D[111] = set([10])
    return D 


def graph_case_11(): 
    D = defaultdict(set) 
    D[0] = set([1,2,3,4]) 
    D[1] = set([0,5,6])
    D[2] = set([0])
    D[3] = set([0])
    D[4] = set([0])
    D[5] = set([0])

    D2 = defaultdict(set) 
    D[7] = set([12,20])
    D[12] = set([38,41])
    D[20] = set([15,64])
    D[38] = set([11])
    D[11] = set([7,31])

    D.update(D2)
    return D 

def graph_case_37(): 

    return defaultdict(set, {0: {2, 19, 8, 14, 31}, \
        1: {0, 32, 4, 6, 7, 8, 26, 15}, 2: {32, 1, 36, 5, 6, 23, 26}, \
        3: {12, 28, 36, 15}, 4: {1, 34, 3, 35, 7, 19, 20, 21, 25, 28, 31}, \
        5: {4, 21, 8, 9, 27, 28, 24}, 6: {2, 3, 7, 10, 11, 12, 22, 26, 30}, \
        7: {17, 26, 10, 4}, 8: {0, 1, 34, 3, 4, 35, 5, 33, 12}, \
        9: {12, 5, 6}, 10: {17, 34}, 11: {10, 36, 6, 7}, 12: {1, 34, 36, 7}, \
        13: {17, 34, 3, 8, 9, 27, 12}, 14: {17, 5}, 15: {34, 19}, \
        16: {0, 1, 3, 35, 5, 27}, 17: {29, 13, 23}, 18: {31, 6, 14, 23}, \
        19: {17, 35, 12}, 20: {0, 10, 6}, 21: {34, 19, 36, 6}, \
        22: {0, 32, 35, 21, 7, 29}, 23: {4, 21, 9, 11, 25, 31}, 24: {3, 28, 14, 31}, \
        25: {23}, 26: {10, 3}, 27: {0, 32, 16, 17, 11, 28, 29}, 28: {13}, \
        29: {3, 19, 25, 26, 28, 14, 31}, 30: {11, 20}, 31: {36, 13}, 32: {34, 19}, \
        33: {26, 3}, 34: {2, 36, 13}, 35: {18, 10}, 36: {25, 7}})