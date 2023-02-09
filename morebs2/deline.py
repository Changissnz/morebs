from morebs2.fit_2n2 import *
from copy import deepcopy

class Delineation:

    def __init__(self):
        self.d = {}
        self.parentIds = []
        self.childIds = []
        return

    def add_edge(self,direction,edge):
        self.d[direction] = edge

    def exclude_edges_from_duplicate_deletion(self,d,l):
        try:
            d.remove(0)
        except:
            pass

        try:
            d.remove(l - 1)
        except:
            pass
        return d

    def remove_duplicates_by_axis(self,axis=0):
        d1,d2 = None,None
        l1,l2 = None,None
        if axis == 0:
            d1,d2 = self.d['l'],self.d['r']
            l1,l2 = len(self.d['l']),len(self.d['r'])
        else:
            d1,d2 = self.d['t'],self.d['b']
            l1,l2 = len(self.d['t']),len(self.d['b'])

        # collect duplicates
        delete1,delete2 = [],[]
        for (i,x) in enumerate(d1):
            ss = np.where((d2 == x).all(axis=1))[0]
            if len(ss) == 0:
                continue
            delete2.extend(ss)
            delete1.append(i)

        # remove indices of endpoints of each edge
        # from deletion
        delete1 = self.exclude_edges_from_duplicate_deletion(delete1,l1)
        delete2 = self.exclude_edges_from_duplicate_deletion(delete2,l2)
        
        d1 = np.delete(d1,delete1,0)
        d2 = np.delete(d2,delete2,0)
        
        if axis == 0:
            self.d['l'],self.d['r'] = d1,d2
        else:
            self.d['t'],self.d['b'] = d1,d2

    def draw_delineation(self):
        return -1

    def visualize_delineation(self):
        return -1

class DLineate22:


    def __init__(self,xyl,clockwise=True):
        assert xyl.shape[1] == 3, "invalid shape"
        assert set(np.unique(xyl[:,2])).issubset({0,1})

        # starting data
        self.xyl = xyl
        # tmp holder
        self.xyl_sorted_x = None
        self.xyl_sorted_y = None 
        # [indices for L0,indices for L1]
        self.lc = None
        # center
        self.c = None 
        # reference point
        self.rp = None
        # bool
        self.clockwise = clockwise
        # label to delineate
        self.label = None
        # label points
        self.lpoints = None
        # label info
        self.linfo = None
        # start point of fit
        self.sp = None
        # container for delineation 
        self.dc = None
        # dc objective
        self.dcobj = None
        # delineation
        self.d = None
        return

    def preprocess(self):
        self.label_counts()
        self.set_target_label()        
        #self.set_start_point()
        self.sort_data() 
        
    def label_counts(self):
        i0 = []
        i1 = []
        for (i,x) in enumerate(self.xyl):
            if x[2] == 0: i0.append(i)
            else: i1.push(i)
        self.lc = [i0,i1] 
        return

    def set_target_label(self,l=None):
        if l == None:
            self.label = 0 if len(self.lc[0]) > len(self.lc[1]) else 1
        else:
            assert l in {0,1} 
            self.label = l

        self.lpoints = deepcopy(self.xyl[self.lc[self.label]])
        self.lpoints = self.lpoints[:,:2]

    def target_label_analysis(self):
        c = self.calculate_center()
        ex = self.calculate_extremum()
        self.linfo = c + ex

    def calculate_center(self):
        '''
        calculates center for the lpoints
        '''
        return [np.mean(self.lpoints[:,0]),np.mean(self.lpoints[:,1])]

    def calculate_extremum(self):
        '''
        calculates extremum for the lpoints
        '''
        # x-min point
        xi = np.argmin(self.lpoints[:,0])
        # x-max point
        xa = np.argmax(self.lpoints[:,0])
        # y-min point
        yi = np.argmin(self.lpoints[:,1])
        # y-max point
        ya = np.argmax(self.lpoints[:,1])
        return [xi,xa,yi,ya]

    def sort_data(self):
        xs = np.argsort(self.xyl[:,0])
        self.xyl_sorted_x = deepcopy(self.xyl[xs])
        ys = np.argsort(self.xyl[:,1])
        self.xyl_sorted_y = deepcopy(self.xyl[ys])
        return

    def collect_break_points(self):
        '''
        clockwise -> [l + t -> increasing order, others in decreasing order]
        counter-clockwise -> [r + b -> increasing order, others in decreasing order]
        '''

        ds = ['l','r','t','b']
        self.d = Delineation()
        for x in ds:
            edge = self.break_points_on_edge(x)
            self.d.add_edge(x,edge)

        # assign directionality
        if self.clockwise:
            self.d.d['r'] = self.d.d['r'][::-1]
            self.d.d['b'] = self.d.d['b'][::-1]
        else:
            self.d.d['l'] = self.d.d['l'][::-1]
            self.d.d['t'] = self.d.d['t'][::-1]

        # remove duplicate points
        self.d.remove_duplicates_by_axis(0)
        self.d.remove_duplicates_by_axis(1)
        return 


    def break_points_on_edge(self,direction):
        '''
        direction := `l` is left edge,`r` is right edge,`t` is top edge,
                     `b` is bottom edge.

                     if `l`|`r`, output is ordered by axis 1. Otherwise, by axis 0. 
        '''
        edge = []

        stat = True
        i = None
        rd = self.xyl_sorted_y if direction in\
            {'l','r'} else self.xyl_sorted_x

        while stat:
            i = self.next_break_point(i,direction)
            if i == None: 
                stat = not stat
                continue
            edge.append(deepcopy(rd[i]))
        return np.array(edge)

    def next_break_point(self,refi,direction):
        rd = None
        axis = None
        if direction in  {'l','r'}:
            rd = self.xyl_sorted_y
            axis = 1
        else:
            rd = self.xyl_sorted_x
            axis = 0
        
        j = None
        if refi == None:
            j = 0
        else:
            # next greater element
            q = rd[refi]
            md = np.inf
            for i in range(refi,len(rd)):
                if rd[i,axis] <= q[axis]:
                    continue
                if rd[i,axis] > q[axis]:
                    if rd[i,axis] - q[axis] < md:
                        md = rd[i,axis] - q[axis]
                        j = i
            if j == None:
                return None
        return self.break_point_tiebreakers(j,direction)

    def break_point_tiebreakers(self,refni,direction):
        #a = 1 if direction in {'l','r'} else 0
        if direction in {'l','r'}: 
            rd = self.xyl_sorted_y
            a = 1
        else:
            rd = self.xyl_sorted_x
            a = 0 

        rp = rd[refni]
        indices = np.where(rd[:,a] == rp[a])

        pp = rd[indices]
        s = None
        if direction == 'r':
            s = np.argmax(pp[:,0])
        elif direction == 'l':
            s = np.argmin(pp[:,0])
        elif direction == 't':
            s = np.argmax(pp[:,1])
        else:
            s = np.argmin(pp[:,1])

        ss = np.where((rd == pp[s]).all(axis=1))
        return ss[0][0]

def test_dataset__Dlineate22_1():
    '''
    x      x
    
    x   x       
    
    x   x      
                x
    x
    
    x      x
    '''
    return np.array([[5.,15.,0],\
        [5.,12.,0],\
        [5.,9.,0],\
        [5.,6.,0],\
        [5.,0.,0],\
        [15.,12.,0],\
        [15.,9.,0],\
        [20.,15.,0],\
        [20.,0.,0],\
        [25.,7.5,0]])    