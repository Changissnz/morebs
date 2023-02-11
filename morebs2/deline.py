from morebs2.fit_2n2 import *
from morebs2.random_generators import *
from collections import defaultdict

class Delineation:

    def __init__(self,label,clockwise,parentIds = [],childIds = []):
        self.d = {}
        self.d_ = None
        self.label = label
        self.clockwise = clockwise
        self.parentIds = parentIds
        self.childIds = childIds
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

            assignIt = self.assign_duplicate_point_to_edge(x,axis)
            
            # case: assign duplicate edge 
            if assignIt == 1:
                delete2.extend(ss)
            else:
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

    def assign_duplicate_point_to_edge(self,p,axis):
        '''
        '''

        if axis == 0:
            k1,k2 = 'l','r'
        else:
            k1,k2 = 't','b'

        x1 = np.mean(abs(self.d[k1][:,axis] - p[axis]))
        x2 = np.mean(abs(self.d[k2][:,axis] - p[axis]))

        if x1 < x2: return 1
        return 2

    def draw_delineation(self):
        '''
        clockwise -> (t,r,b,l)
        counter-clockwise -> (t,l,b,r)
        '''
        self.d_ = []

        one = self.point_sequence_to_curve_set(self.d['t'],'t')
        
        if self.clockwise:
            two = self.point_sequence_to_curve_set(self.d['r'],'r')
            four = self.point_sequence_to_curve_set(self.d['l'],'l')
        else:
            four = self.point_sequence_to_curve_set(self.d['r'],'r')
            two = self.point_sequence_to_curve_set(self.d['l'],'l')

        three = self.point_sequence_to_curve_set(self.d['b'],'b')

        self.d_.extend(one)
        self.d_.extend(two)
        self.d_.extend(three)
        self.d_.extend(four)

    def point_pair_to_curve(self,p1,p2,ad):
        # case: line
        if p1[0] == p2[0] or\
            p1[1] == p2[1]:
            c = Line([p1,p2]) 
        else:
        # case: default is LogFit22
            direction = np.argsort(np.array([p1,p2])[:,1])
            c = LogFit22(np.array([p1,p2]),direction = direction)
        return DCurve(c,ad) 

    def point_sequence_to_curve_set(self,ps,ad):
        l = len(ps) - 1
        cs = []
        for i in range(0,l):
            cs.append(self.point_pair_to_curve(\
                deepcopy(ps[i]),deepcopy(ps[i+1]),ad))
        return cs

    # TODO: test
    def classify_point(self,p):
        '''
        return := label if p in delineation, otherwise -1
        '''

        x1,x2 = self.point_to_relevant_curvepair(p) 
        if x1 == None:
            return -1
            
        if x1.is_ap(p) and x2.is_ap(p):
            return self.label
        return -1

    def point_to_relevant_curvepair(self,p):
        '''
        searches for a curve pair (left&right)|(top&bottom)
        that 
        '''
        l,r,t,b = None,None,None,None

        for x in self.d_:
            if x.in_point_range(p):
                if x.ad == 'l': l = x
                elif x.ad == 'r': r = x
                elif x.ad == 't': t = x
                elif x.ad == 'b': b = x

        if type(l) == type(None) or type(r) == type(None):
            if type(t) == type(None) or type(b) == type(None):
                return None,None
            return t,b
        return l,r

    def classify_point_by_curveset(self,p,cs):
        for c in cs:
            if not c.in_point_range(p):
                continue

            if not c.is_ap(p):
                return -1
        return self.label

    def complement_set_to_curve(self,c):
        '''
        collects all curves in delineation with an 
        activation direction parallel to that of c
        that have a range that intersects that of c. 

        * note: method can only be called after `delineation_to_initial_estimate`. 
        '''
        axis = 1 if c.ad in {'l','r'} else 0
        pr = c.point_range()[axis]

        if c.ad == 'l':
            x = 'r'
        elif c.ad == 'r':
            x = 'l'
        elif c.ad == 'b':
            x = 't'
        else:
            x = 'b'

        q = []
        for c_ in self.d_:
            if c_.ad != x: continue
            pr2 = c_.point_range()[axis]
            if pr2[0] >= pr[0] and pr2[0] <= pr[1]:
                q.append(deepcopy(c_))
            elif pr2[1] >= pr[0] and pr2[1] <= pr[1]:
                q.append(deepcopy(c_))
        return q

    def visualize_delineation(self):
        ps = []
        for c in self.d_:
            l = c.form_point_sequence()
            ps.extend(l)
        ps = np.array(ps)
        basic_2d_scatterplot(ps[:,0],ps[:,1],c='b')
        return

class DLineate22:

    def __init__(self,xyl,clockwise=True):
        assert xyl.shape[1] == 3, "invalid shape"
        assert set(np.unique(xyl[:,2])).issubset({0,1})
        # starting data
        self.xyl = xyl
        # tmp holder
        self.xyl_sorted_x = None
        self.xyl_sorted_y = None 
        # labels -> label indices
        self.lc = None
        # labels -> excluded indices; used in the case of 
        #           delineations of degree 2 and up (nested delineations)
        self.lc_exclude = None
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
        # delineation
        self.d = None

        # container for finalized delineations
        self.ds = [] 
        return

    ############# preprocessing methods

    def preprocess(self):
        self.label_counts()
        self.set_target_label()        
        #self.set_start_point()
        self.sort_data() 
        
    def label_counts(self):
        self.lc = defaultdict(list)
        for (i,x) in enumerate(self.xyl):
            self.lc[x[2]].append(i)
        return

    def set_target_label(self,l=None):
        # default: choose min label
        if l == None:
            x = np.array([(k,len(v)) for (k,v) in self.lc.items()])
            a = np.argmin(x[:,1])
            self.label = x[a,0]
        else:
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
        xs = np.argsort(self.lpoints[:,0])
        self.xyl_sorted_x = deepcopy(self.xyl[xs])
        ys = np.argsort(self.lpoints[:,1])
        self.xyl_sorted_y = deepcopy(self.xyl[ys])
        return

    ################## initial delineation

    def collect_break_points(self,pi=[],ci=[]):
        '''
        clockwise -> [l + t -> increasing order, others in decreasing order]
        counter-clockwise -> [r + b -> increasing order, others in decreasing order]
        '''

        ds = ['l','r','t','b']
        self.d = Delineation(label=self.label,clockwise=self.clockwise,parentIds=pi,childIds=ci)
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

        self.d.draw_delineation()
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
            edge.append(deepcopy(rd[i,:2]))
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

    ######### start: delineation optimizer

    def optimize_delineation(self):
        s = 0 
        for (i,x) in enumerate(self.d.d_):
            s += self.improve_curve(x)
        return s

    def improve_curve(self,c):
        # get pertinent points to curve
        pp = self.pertinent_points_to_curve(c)
        # get curveset
        cs = self.d.complement_set_to_curve(c) + [c]

        # classify points
        s1 = self.classify_pertinent_points(cs,pp)
        # alternate and reclassify
        c.modulate_fit()
        s2 = self.classify_pertinent_points(cs,pp)
        if s1 > s2:
            c.modulate_fit()
            return 0 
        return s2 - s1

    def classify_pertinent_points(self,cs,pp):

        s = 0
        for p in pp:
            l = self.d.classify_point_by_curveset(p[:2],cs)
            if l != -1:
                s += 1
        return s

    def pertinent_points_to_curve(self,c):
        ps = []
        for p in self.xyl:
            if c.in_point_range(p):
                ps.append(p)
        return ps 

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

def test_dataset__Dlineate22_1_v2():
    d = test_dataset__Dlineate22_1()
    t2 = generate_random_xyl_points_at_center(\
        [12.,8.55,5],[[0.,5.,5],[15.,25.,8]],1)
    return np.vstack((d,t2))


def test_dataset__Dlineate22_1_v3(numPoints,rdistance):
    data = test_dataset__Dlineate22_1()
    dl = DLineate22(data)
    dl.preprocess()
    dl.collect_break_points()

    l = len(dl.d.d_)
    ps = 0
    points = []
    i = 0
    while ps < numPoints:
        a = random.random()
        if a > 0.5:
            pr = dl.d.d_[i].get_point()
            rp = random_point_near_2d_point_pair(\
                pr,rdistance)
            rp_ = [rp[0],rp[1],1.]
            points.append(rp_)
            ps += 1
        i = (i + 1) % l
    points = np.array(points)
    return np.vstack((data,points))