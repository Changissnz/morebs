from morebs2.fit_2n2 import *
from morebs2.line import * 
from morebs2.matrix_methods import * 
from copy import deepcopy
#import numpy as np

class D22Obj:

    def __init__(self,obj,clockwise):
        # shuffle({l|r,u,d})
        self.obj = obj
        self.clockwise = clockwise
        self.mod_count = 0  
        return

    def modulate(self):
        if self.obj == ['u','l']:
            self.obj = ['r','u']
        elif self.obj == ['r','u']:
            self.obj = ['d','r']
        elif self.obj == ['d','r']:
            self.obj = ['l','d']
        elif self.obj == ['l','d']:
            self.obj = ['u','l']
        elif self.obj == ['l','u']:
            self.obj = ['d','l']
        elif self.obj == ['d','l']:
            self.obj = ['r','d']
        elif self.obj == ['r','d']:
            self.obj = ['u','r']
        else:
            self.obj = ['l','u']
        self.mod_count += 1
        return

class DCurve:

    def __init__(self,fitstruct,activationDirection):
        self.fs = fitstruct
        self.ad = activationDirection

    def x_given_y(self,y):
        if type(self.fs) == Line:
            return self.fs.x_given_y(y)
        return self.fs.g(y)

    def y_given_x(self,x):
        if type(self.fs) == Line:
            return self.fs.y_given_x(y)
        return self.fs.f(y)

    def __str__(self):
        l = None
        if type(self.fs) == Line: l = "line"
        elif type(self.fs) == LogFit22: l = "logfit22"
        else: l = "exp2fit22"
        p = self.get_point()

        s = "* struct: {}\n".format(l)
        s2 = "* points\n-- {}\n-- {}\n".format(p[0],p[1])
        s3 = "* direction: {}\n".format(self.ad)
        return s + s2 + s3 + "---"

    def get_point(self):
        if type(self.fs) == Line: 
            return deepcopy(self.fs.endpoints)
        return deepcopy(self.fs.ps)

    def point_range(self):
        '''
        [[x-min,x-max],[y-min,y-max]]
        '''
        p = deepcopy(self.fs.endpoints) if type(self.fs) == Line\
            else deepcopy(self.fs.ps)

        x = np.sort(p[:,0])
        y = np.sort(p[:,1])
        return np.array([x,y])

    def is_ap(self,p):
        '''
        is activation point?
        '''
        if self.ad in {'l','r'}:
            p2 = self.y_given_x(p[0])
            if self.ad == 'l':
                return p2 >= p[1]
            return p2 <= p[1]
        else:
            p2 = self.x_given_y(p[1])
            if self.ad == 'd':
                return p2 >= p[0]
            return p2 <= p[0]

    def modulate_fit(self):
        if type(self.fs) == Line:
            return

        if type(self.fs) == LogFit22:
            c = LogFit22(deepcopy(self.fs.ps))
        else:
            c = Exp2Fit22(deepcopy(self.fs.ps))
        self.fs = c


class Delineate22Container:

    def __init__(self,xy,clockwise=True):
        assert clockwise in {True,False}
        self.xy = xy
        self.ref = None
        self.restriction = None
        self.clockwise = clockwise
        self.cp = np.empty((0,2))
        self.readd = [] 

    def set_ref(self,ri):
        r = deepcopy(self.xy[ri])
        self.ref = r
        self.xy = np.delete(self.xy,ri,0)

    def readd_points(self):
        if self.readd != []: 
            self.xy = np.vstack((self.xy,np.array(self.readd)))
        self.readd = []

    def next_point(self,objective):
        if type(self.ref) == type(None):
            return None

        self.objective_restriction(objective)

        # get the minumum point distance
        return self.minumum_distance_to_reference(objective)

    def minumum_distance_to_reference(self,objective):
        '''
        collects all candidate points of minumum distance to reference point
        based on objective. Then delete all those candidate points and output
        the best one.  
        '''
        candidates = []
        d = np.inf

        # get nearest point candidates
        for (i,p2) in enumerate(self.xy):
            if not self.restriction(p2):
                continue

            d2 = self.directional_distance(p2,objective[0])
            if d2 < d:
                candidates = [i]
                d = d2
            elif d2 == d and d != np.inf:
                candidates.append(i)
                
        # analyze candidates
            # case: more than one candidate, choose the one with min
        return self.process_candidates(candidates,objective)

    def process_candidates(self,c,objective):
        # case: no remainining candidates, return message to
        #       modulate objective
        if len(c) == 0:
            return None

        # case: only one point
        if len(c) == 1:
            r = deepcopy(self.xy[c[0]])
            self.xy = np.delete(self.xy,c[0],0)
            return r

        # case: more than one point
            # sort points
        s = deepcopy(self.xy[c])
        c = np.array(c)
        c = c.reshape((len(c),1))
        s = np.hstack((s,c))

        if objective[1] in {'l','r'}:
            indices = np.argsort(s[:,0])         
        else:
            indices = np.argsort(s[:,1])

        s = s[indices]
        if objective[1] in {'l','d'}:
            s = s[::-1]

        # ignore the last point
        rem = deepcopy(s[-1])
        q = deepcopy(s[0])
        self.readd.append(rem[:2])

        # delete all points except for the extreme opposite
        self.xy = np.delete(self.xy, np.asarray(s[:,2],dtype='int'),0)
        return q[:2]

    def objective_restriction(self,objective):
        # case: vertical
        if objective[0] in {'u','d'}: 
            if objective[0] == 'u':
                self.restriction = lambda p: p[1] >= self.ref[1]
            else:
                self.restriction = lambda p: p[1] <= self.ref[1]
        else:
        # case: horizontal
            if objective[0] == 'l':
                self.restriction = lambda p: p[0] >= self.ref[0]
            else:
                self.restriction = lambda p: p[0] <= self.ref[0]

    def directional_distance(self,p,direction):
        # case: p is the same point as ref, distance is inf
        if equal_iterables(p,self.ref):
            return np.inf

        if direction in {'l','r'}:
            x = abs(p[0] - self.ref[0])
        else: 
            x = abs(p[1] - self.ref[1])
        return x if x != 0. else np.inf

class Delineate22:
    '''
    works only with binary labels. 
    '''

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
        
        self.target_label_analysis() 
        self.set_start_point()
        
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
        return

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

    def set_start_point(self):
        '''
        start point is right-most by default; used for
        terminating condition.
        '''
        self.dc = Delineate22Container(deepcopy(self.lpoints),self.clockwise)
        if self.clockwise:
            obj = ['d','l']
            if self.lpoints[self.linfo[3]][0] == self.lpoints[self.linfo[4]][0]:
                self.dc.set_ref(self.linfo[4])
            else:
                self.dc.set_ref(self.linfo[3])
        else: 
            obj = ['u','r']
            if self.lpoints[self.linfo[3]][0] == self.lpoints[self.linfo[5]][0]:
                self.dc.set_ref(self.linfo[5])
            else:
                self.dc.set_ref(self.linfo[3])

        self.sp = deepcopy(self.dc.ref)
        self.rp = deepcopy(self.sp)
        self.dcobj = D22Obj(obj,self.clockwise) 

    ##########################################

    def collect_break_points(self):
        '''
        main method
        '''
        self.d = []

        for i in range(5):
            ps = self.break_points_on_edge()
            self.dcobj.modulate()
            if ps == []: break 
            self.d.append(ps)

        # reset the dcobj
        self.reset_dcobj()

    def break_points_on_edge(self):
        '''
        collects critical points on one edge
        '''
        # ERROR: check for p
        #bc = self.break_condition_for_fetch()
        ps = deepcopy(self.rp)
        if type(ps) == type(None): return [] 
        c = []
        while True:
            c.append(deepcopy(ps))
            ps = self.dc.next_point(self.dcobj.obj)
            if type(ps) == type(None): break
            self.dc.ref = deepcopy(ps)
            self.rp = deepcopy(ps)

        # add pertinent points back to cache
        self.dc.readd_points()
        return c
    
    def sort_data(self):
        xs = np.argsort(self.xyl[:,0])
        self.xyl_sorted_x = deepcopy(self.xyl[xs])
        ys = np.argsort(self.xyl[:,1])
        self.xyl_sorted_x = deepcopy(self.xyl[ys])
        return

    def break_condition_for_fetch(self):

        if self.dcobj.mod_count == 4:
            i = 0 if self.dcobj.obj[1] in {'l','r'} else 1
            break_condition = lambda p: p[self.dcobj.obj[1]] == self.sp[i]
        else: 
            break_condition = lambda p: True
        return break_condition

    def pertinent_points_to_curve(self,c):
        '''
        for c.ad:
        - l,r -> all points.x in range(curve.x)
        - u,d -> all points.y in range(curve.y)
        '''

        # scan sorted-by-x
        indices = []
        pts = c.point_range()
        i = 0 if c.ad in {'l','r'} else 1 
        f = lambda p: p >= pts[i,0] and p <= pts[i,1]
        indices.append(i) 
        s,e = None,None
        for (j,p) in enumerate(self.xyl_sorted_x):
            if f(p):
                if s == None: s = j
                else: e = j
        return [i,s,e]

    def improve_delineation(self):
        '''
        improviaanos
        '''

        for c in self.d:
            self.improve_curve(c)
    
    def improve_curve(self,c):
        '''
        attempts to improve score of curve based on relevant
        points by modulating it.
        '''

        pp = self.pertinent_points_to_curve(c)
        cs = self.complement_set_to_curve(c)

        s1 = self.score_curve_(c,cs,pp)

        c.modulate_fit()
        s2 = self.score_curve_(c,cs,pp)

        if s1 > s2:
            c.modulate_fit()
            return s1
        return s2

    def score_curve_(self,cr,cs,pp):
        '''
        oscillates the curve
        '''

        def complement_for_point(cs,p):
            for c in cs: 
                pr = c.point_range()

                # check in y-range
                if c.ad in {'l','r'}:
                    if p[1] >= pr[0,0] and p[1] <= pr[0,1]:
                        return c
                # check in x-range
                else:
                    if p[1] >= pr[1,0] and p[1] <= pr[1,1]:
                        return c
            return None

        s = 0
        for p in pp:
            c = self.complement_for_point(cs,p)
            l = self.label_of_point_by_curvepair(p,[cr,c])
            if l == p[2]:
                s += 1
        return s

    def label_of_point_by_curvepair(self,p,cp):
        r = int(cp[0].is_ap(p[:2]) and cp[1].is_ap(p[:2]))
        if self.label == 0:
            return (r + 1) % 2
        return r

    def opposing_curve_pair_to_point(self,p,rc):
        '''
        collects all curves in delineation with an 
        activation direction perpendicular to that of c
        that have a range that intersects that of c on
        the relevant axis.

        * note: method can only be called after `delineation_to_initial_estimate`. 
        ----------------------
        p := point
        rc := reference curve
        '''

        c1,c2 = None,None
        # consider ad in {l,r}
        directions = {'u','d'} if rc.ad in {'l','r'} else {'l','r'}
        axis = 0 if c.ad in {'l','r'} else 1
        for c_ in self.d:
            pr2 = c_.point_range()[axis]
            if p[axis] >= pr2[0] and p[axis] <= pr2[1]:
                if type(c1) == type(None):
                    c1 = deepcopy(c_)
                else:
                    c2 = deepcopy(c_)
        return c1,c2

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
        elif c.ad == 'd':
            x = 'u'
        else:
            x = 'd'

        q = []
        for c_ in self.d:
            if c_.ad != x: continue
            pr2 = c_.point_range()[axis]
            if pr2[0] >= pr[0] and pr2[0] <= pr[1]:
                q.append(deepcopy(c_))
            elif pr2[1] >= pr[0] and pr2[1] <= pr[1]:
                q.append(deepcopy(c_))
        return q

    def pertinent_curves_on_edge(self,p,ad):
        '''
        '''
        q = []
        for c_ in self.d:
            if c_.ad != ad: continue
            pr2 = c.point_range()[axis]
            if pr2[0] >= pr[0] and pr2[0] <= pr[1]:
                q.append(deepcopy(c_))
            elif pr2[1] >= pr[0] and pr2[1] <= pr[1]:
                q.append(deepcopy(c_))
        return -1

    def is_point_relevant_to_curve(self,c,p):
        '''
        '''
        # all points in y-range if c.ad l|r
        axis = 1 if c.ad in {'l','r'} else 0
        pr = c.point_range()[axis]
        return p[axis] >= pr[0] and p[axis] <= pr[1]

    def cross_check(self,p):
        '''
        determines the label of point p
        '''
        return -1

    def delineation_to_initial_estimate(self):
        self.reset_dcobj()
        #
        d = []
        while len(self.d) > 0:
            d_ = self.d.pop(0)
            css = self.point_sequence_to_curve_set(d_)
            self.dcobj.modulate()

            if len(css) == 0:
                continue  
            d.extend(css)

        # connect the last point to the first point
        sp = d[0].get_point()
        ep = d[-1].get_point()
        if not equal_iterables(sp[0],ep[1]):
            css = self.point_pair_to_curve(ep[1],sp[0],d[-1].ad)
            d.append(css)
        self.d = d
        return

    def point_pair_to_curve(self,p1,p2,ad):
        # case: line
        if p1[0] == p1[0] or\
            p1[1] == p1[1]:
            c = Line([p1,p2]) 
        else:
        # case: default is LogFit22
            c = LogFit22(np.array([p1,p2]))
        return DCurve(c,ad) 

    def point_sequence_to_curve_set(self,ps):
        '''
        activation of curve is dc.objective[1]
        '''
        l = len(ps) - 1
        cs = []
        for i in range(0,l):
            cs.append(self.point_pair_to_curve(\
                deepcopy(ps[i]),deepcopy(ps[i+1]),self.dcobj.obj[1]))
        return cs

    def reset_dcobj(self):
        while self.dcobj.mod_count % 4:
            self.dcobj.modulate()
        



# gravity_swing!

###### TODO: make test cases. 

def test_dataset__Delineate22_1():
    '''
    x      x
    x   x       x
    x   x      
    x
    x      x

    ---------------
                x


    x      x
    x   x
    x   x      
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

# d,l => [25,7.5],[5,6],[20,0]
# l,u => [20,0],[15,9],[5,0]
# u,r => [5,0],[15,12]
# 
def test_dataset__Delineate22_2():
    '''
        x  

    x   

        x
    '''
    return np.array([[50.,70.,0],\
        [90.,100.0,0],\
        [90.,40.,0]])

def test_dataset__Delineate22_3():
    '''
    pre-defined dataset
    '''
    f = x_effector_function_type_modulo(xyi_op_type_1,[200.,600.])
    return generate_2d_data_from_function__multipleY(multiple_y_function_x_effector,[-20.,40.],0.25,additionalArgs = (f,[0,3]))

def test_dataset__Delineate22_4():
    '''
    x    x

    x    x
    '''

    return -1

def test_dataset__Delineate22_5():
    '''
    
    xx
    x
    xxx
    x x x x 
   x  x
     x x 
    xx
  xxxxxx
    '''
    return -1




### TODO: relocate
def test__Delineate22__collect_break_points__case_1():

    # case one
    xyl = test_dataset__Delineate22_1()
    d22 = Delineate22(xyl,clockwise=True)
    d22.preprocess()
    d22.collect_break_points()

    sol = [[np.array([25. ,  7.5]), np.array([5., 6.]), np.array([20.,  0.])],\
    [np.array([20.,  0.]), np.array([15.,  9.]), np.array([5., 0.])],\
    [np.array([5., 0.]), np.array([15., 12.]), np.array([ 5., 15.])],\
    [np.array([ 5., 15.]), np.array([20., 15.])],\
    [np.array([20., 15.])]]
    for (j,d_) in enumerate(d22.d):
        for (i,d2) in enumerate(d_):
            assert equal_iterables(d2,sol[j][i])
    return d22

def test__Delineate22__collect_break_points__case_2():
    xyl = test_dataset__Delineate22_2()
    d22 = Delineate22(xyl,clockwise=True)
    d22.preprocess()
    d22.collect_break_points()
    return d22 