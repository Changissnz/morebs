from morebs2.numerical_space_data_generator import * 

def OneDimClassifier_test_dataset_1(): 

    bounds = np.zeros((4,2)) 
    bounds[:,0] -= 10 
    bounds[:,1] += 10 

    startPoint = np.zeros((4,)) - 10 
    columnOrder = np.array([0,1,2,3]) 
    ssi_hop = 3 

    rch2 = sample_rch_1_with_update(deepcopy(bounds),deepcopy(bounds),ssi_hop,0.1)
    cv = 0.4
    rssi2 = ResplattingSearchSpaceIterator(bounds, startPoint, \
        columnOrder, SSIHop = ssi_hop,resplattingMode = ("relevance zoom",rch2), additionalUpdateArgs = (cv,))

    ##xr = np.array([next(rssi) for _ in range(1000)]) 
    xr2 = np.array([next(rssi2) for _ in range(1000)])
    return xr2 