from morebs2.search_space_iterator import *
from morebs2.relevance_functions import *
from morebs2.relevance_functions_extended import decimal_places_of_float

from morebs2.numerical_space_data_generator import one_random_noise_,NSDataInstructions
from morebs2.hop_pattern import vector_hop_in_bounds
from morebs2.poly_struct import *

from morebs2.poly_factor import *

from morebs2.numerical_extras import * 

from morebs2.distributions import *
from morebs2.fit_2n2 import *
from morebs2.deline import *
from morebs2.deline_mc import * 
from morebs2.deline_helpers import * 

from morebs2.modular_labeller import *

from morebs2.matrix_methods import * 

def gen_t5(): 
    random.seed(1) 
    bounds = np.array([[0,27],[-9,9],[-12,0],[12,72],[50,74]])

    sp = np.copy(bounds[:,0])
    columnOrder = None
    ssih = 6
    bInf = (bounds,sp,columnOrder,ssih,())

    # make the label delta pattern and rch
    tickLength = (6 ** 4) 
    deviationPr = 0.02
    deviationRange = [-2,2]
    labelRange = [0,6]
    startLabel = 2
    ldp = LabelDeltaPattern(tickLength, deviationPr, deviationRange, labelRange,startLabel)

    rch = rch_modular_labeller_1(ldp)
    rm = ("relevance zoom",sample_rch_blind_accept())

    nsdi = NSDataInstructions(bInf, rm,"t5.csv",'w',noiseRange = None,writeOutMode = rch)
    nsdi.make_rssi()

    c = 0
    while nsdi.fp and c < 15:
        nsdi.next_batch_()
        c += 1
    return 

def gen_t4():
    s4 = test_dataset__DLineateMC_1()
    write_vector_sequence_to_file("t4.csv",s4,'w')
    return 