from collections import defaultdict
from .line import *
import random
from copy import  deepcopy
import operator

# NOTE: when devising euclidean_point_distance measures
'''
reference count
vector -> ed vector -> (ed in bounds):float -> bool(float)

reference existence
vector -> ed vector -> (ed in bounds):bool
'''
lambda_floatin = lambda x,b: x >= min(b) and x <= max(b)

lambda_pointinbounds = lambda p,b: point_in_bounds_(b,p)

def lambda_countpointsinbounds(p,b):
    if b.shape[0] == 1:
        mask = np.logical_and(p >= b[0,0],p <= b[0,1])
    else:
        mask = np.logical_and(p >= b[:,0],p <= b[:,1])
    q = len(np.where(mask == True)[0])
    return q

def lambda_ratiopointsinbounds(p,b):
    x = lambda_countpointsinbounds(p,b)
    return zero_div(x,len(p),np.inf)

def random_select_k_unique_from_sequence(s, k):
    s = list(s)
    assert len(s) >= k
    random.shuffle(s)
    return s[:k]


################################### TODO: delete these functions after refactor

# TODO:
def relevance_zoom_func_1(referencePoint,boundsDistance,activationThreshold):
    # TODO: check for 0-case
    assert boundsDistance >= 0.0, "invalid bounds distance {}".format(boundsDistance)
    assert activationThreshold >= 0.0 and activationThreshold <= 1.0, ""
    return lambda p: euclidean_point_distance(p, referencePoint) <= activationThreshold * boundsDistance

def relevance_zoom_func_2(referencePoint, modulo, activationThreshold):
    rp = np.array(referencePoint, dtype = "int")
    lim = referencePoint.shape[0]
    def p(x):
        q = (rp - np.array(x,dtype="int")) % 2
        return lambda p: True if len(q == 1.0) >= lim else False
    return p


## TODO: delete?
def relevance_func_2(modulo, moduloPercentileRange):
    '''
    a sample relevance function to help demonstrate the work of CenterResplat.

    :return: function(vector) -> True if all values in (point % modulo) fall within moduloPercentileRange
    :rtype: function<vector>
    '''
    assert modulo >= 0.0, "invalid modulo"
    assert is_valid_point(moduloPercentileRange)
    assert min(moduloPercentileRange) >= 0.0 and max(moduloPercentileRange) <= 1.0
    assert moduloPercentileRange[0] <= moduloPercentileRange[1]

    minumum,maximum = moduloPercentileRange[0] * modulo,moduloPercentileRange[1] * modulo

    def f(p):
        p_ = p % modulo
        return np.all(p_ >= minumum) and np.all(p_ <= maximum)

    return f

def vector_modulo_function_with_addon(modulo, addOn):
    """
    boolean function, addon determines
    """
    def x(v):
        q = np.array(v,dtype="int")
        v_ = q % modulo
        return addOn(v_)
    return x

##### add-on functions : vector -> bool

"""
"""
def vf_vector_reference(vr, pw):
    def x(v):
        return pw(vr,v)
    return x

def subvector_iselector(indices):
    def a(v):
        return v[indices]
    return a


def m(v,addOn,iov,outputType):
    """
    m is index|value selector function for arg. to func<addOn>
    """
    assert iov in [0,1,2]

    x = []
    for t in enumerate(v):
        qi = t[iov] if iov != 2 else t
        if addOn(qi):
            if outputType == 1: q = t[1]
            else: q = t
            x.append(q)
    return np.array(x)

def subvector_selector(addOn, inputType =1, outputType = 1):
    """
    outputs elements by addOn

    :param addOn: function that outputs either singleton or pair
    :type addOn: function
    """
    assert inputType in [0,1,2], "invalid input"
    assert outputType in [1,2], "invalid output"

    def m_(v):
        return m(v,addOn,inputType,outputType)

    return m_

## $
def addon_singleton__bool__criteria_range_each(rangeReq):
    '''
    :param rangeReq: proper bounds vector
    :type rangeReq: proper bounds vector
    :return:
    :rtype: function<vector>->bool
    '''

    assert is_proper_bounds_vector(rangeReq), "invalid ranges"
    # rangeReq := length 1 or vr.shape[0]
    if rangeReq.shape == (1,2):
        q = rangeReq[0]
        p = lambda v: np.all(v >= q[0]) and np.all(v <= q[1])
    else:
        p = lambda v: point_in_bounds(rangeReq,v)
    return p

def addon_singleton__bool__criteria_distance_from_reference(rf, dm, dt,cf):
    """
    :param rf: reference value
    :type rf: ?
    :param dm: distance measure between (rf,v)
    :type dm: func<v1::vector,v2::vector>->float
    :param dt: distance threshold
    :type dt: float
    :param cf: comparator function on (dist,dt)
    :type cf: function<?>->?
    :return: function on `cf`
    :type: function<cf>->?
    """
    return lambda v: cf(dm(rf,v),dt)

class RCInst:
    """
    class that acts as a node-like structure and a function, node is designed to be used w/ either
    
    (1) referential data (from .outside of chain)
    
    (2) (standard operator,operand)

    This class is a function on an argument `x` that works by one of the two paths:
    (1) deciding path:

    (2) non-deciding path: 

    :attribute rf: reference value, as argument to dm(v,rf)
    :attribute dm: function f(v,rf)
    :attribute cf: function f(v,*dt), examples include  operator.lt, operator.le, lambda_floatin, np.cross
    :attribute dt: value, use in the case of decision
    """

    def __init__(self):
        self.rf = None
        self.dm = None
        self.cf = None
        self.dt = None
        self.updateFunc = {}
        self.updateInfo = None
        self.updatePath = {} # k = index -> function argument indices in updateInfo

    def inst_update_var(self):

        for k,v in self.updatePath.items():

            # fetch function args
            q = []
            for v_ in v:
                q.append(self.updateInfo[v_])

            f = self.updateFunc[k]
            x = f(*tuple(q))
            self.update_var(k,x)

    def load_update_info(self,updateInfo):
        self.updateInfo = updateInfo

    def update_var(self,k,v):
        '''
        updates one of the class attributes `rf`,`dm`,`cf`,or `dt`
        with value `v`.

        :param k: attribute name
        :type k: str
        :param v: new update value
        :type v: ?
        '''
        if k == "rf":
            self.rf = v
        elif k == "dm":
            self.dm = v
        elif k == "cf":
            self.cf = v
        elif k == "dt":
            self.dt = v
        else:
            raise ValueError("invalid key _{}_".format(k))

    def mod_cf(self,dcf):
        self.cf = dcf(self.cf)
        return

    def set_reference_data(self,rf,dm):
        self.load_var_ref(rf)
        self.load_var_dm(dm)
        return

    ############# some functions

    def branch_at(self,n,i):
        return -1

    def load_var_ref(self,rf):
        self.rf = rf

    def load_var_cf(self,cf):
        self.cf = cf

    def load_var_dt(self,dt):
        """
        threshold variable, use as cf(v,dt)
        """
        self.dt = dt

    def path_type(self):
        if type(self.rf) != type(None):
            return "ref"
        return "dir"

    # TODO: class does not know.
    def output_type():
        return -1

    def load_var_dm(self,dm):
        """
        :param dm: function on (rf,v)
        """
        self.dm = dm

    def load_path(self):
        # deciding path
        if type(self.dt) != type(None):
            # for output type bool|float
            if self.path_type() == "ref":
                # calculates distance from .reference
                self.f = lambda v: self.cf(self.dm(self.rf,v),self.dt)#*self.ct,self.dt)
            else:
                self.f = lambda v: self.cf(v, self.dt)#*self.ct,self.dt) # *self.ct
        # non-deciding path
        else:
            if self.path_type() == "ref":
                # calculates distance from .reference
                self.f = lambda v: self.cf(self.dm(self.rf,v))#,*self.ct)
            else:
                self.f = lambda v: self.cf(v)#,*self.ct)
        return deepcopy(self.f)

class RChainHead:
    """
    RChainHead is a node-like structure

    :param s: the chain of `RCInst` instances
    :type s: list(`RCInst`)
    :param vpath: the sequence of transformation values that a value `v` goes through at each node in
    `s`. 
    :type vpath: list(values) 
    :param updatePath: 
    :type updatePath:
    """
    
    def __init__(self):
        self.s = []
        self.vpath = []
        self.updatePath = {} # node index -> update indices

    def update_rch(self):
        for s_ in self.s:
            s_.inst_update_var()

    def load_update_path(self,up):
        '''
        loads an update_path
        '''
        self.updatePath = up
        return

    def load_update_vars(self,varList):
        for k,v in self.updatePath.items():
            vl = []
            for (i,v2) in enumerate(varList):
                if i in set(v):
                    vl.append(v2)
            self.s[k].load_update_info(vl)

    def link_rch(self,rch,linkerFunc, prev = False):
        if prev:
            return linkerFunc(self,rch)
        return linkerFunc(self,rch)

    def vpath_subset(self,si):
        return [x for (i2,x) in enumerate(self.vpath) if i2 in si]

    def load_cf_(self, rci,cfq):
        '''

        '''
        if type(cfq) == type(()):
            xs = tuple(self.vpath_subset(cfq[1]))
            cf = cfq[0](*xs)

            # below method
            rci.load_var_cf(cf)
        else:
            rci.load_var_cf(cfq)

    def make_node(self,kwargz):
        """
        instantiates an `RCInst` node using the argument
        sequence `kwargz`.
        Note: selectorIndices refer to values in `vpath`.

        :param kwargz: If index 0 is `r` (node uses reference values), then
                format is (`r`,rf,dm|(dm,selector indices),cf|(cf,selectorIndices),?dt?).
                If index 0 is `nr` (node does use reference values), then
                format is (`nr`,cf|(cf,selectorIndices),?dt?).
                
                Please see the description for `RCInst` for details on these values.
        :type kwargz: iterable
        """
        assert kwargz[0] in ["r","r+","nr"]

        rci = RCInst()
        if kwargz[0] == "r":
            assert len(kwargz) in [4,5], "invalid length for kwargs"
            rci.set_reference_data(kwargz[1],kwargz[2])
            self.load_cf_(rci,kwargz[3])
            try: rci.load_var_dt(kwargz[4])
            except: pass

        elif kwargz[0] == "nr":
            assert len(kwargz) in [2,3], "invalid length for kwargs"
            self.load_cf_(rci,kwargz[1])
            try: rci.load_var_dt(kwargz[2])
            except: pass

        elif kwargz[0] == "r+":
            return -1

        rci.load_path()
        return rci

    def add_node_at(self, kwargz, index = -1):
        assert index >= -1, "invalid index"
        n = self.make_node(kwargz)
        if index == -1:
            self.s.append(n)
        else:
            self.s.insert(index,n)
        return -1

    def apply(self,v):
        '''
        main function of RChainHead; applies the composite function (full function path) onto v
        
        :param v: argument into chain function
        :type v: ?
        '''

        i = 0
        v_ = np.copy(v)
        self.vpath = [v_]

        while i < len(self.s):
            q = self.s[i]
            v_ = q.f(v_)
            self.vpath.append(v_)
            i += 1
        return v_

    def cross_check(self):
        return -1

    def merge(self):
        return -1

    def __next__(self):
        return -1

####----------------------------------------------------------

###### START: helper functions for next section

def boolies(v_):
    return v_ == True

def column_selector(columns, flatten = False):

    def p(v):
        q = v[:,columns]
        if flatten: return q.flatten()
        return q
    return p

def vector_index_selector(indices):
    def p(v):
        return v[indices.astype('int')]
    return p

def vector_index_inverse_selector(v):
    def p(indices):
        return v[indices.astype('int')]
    return p


###### END: helper functions for next section

###### START: functions used for relevance zoom

'''
'''
def RCHF__point_in_bounds(b):
    kwargs = ['nr', lambda_pointinbounds, b]
    # f : v -> v::bool
    rc = RChainHead()
    rc.add_node_at(kwargs)
    return rc.apply

def hops_to_default_noise_range(h):
    return np.array([[(h ** -1) / 2.7, (h ** -1) / 2.3]])

#################################### start : ostracio && deletira

"""

arguments:
- k :=
- h := hop value of SSI

return:
- matrix, dim (m,k), m the required number of points
"""
def hops_to_coverage_points__standard(k,h):

    z = np.zeros((k,))
    o = np.ones((k,))

    #
    partition = n_partition_for_bound(np.array([z,o]).T,h)

    # case: odd
    if h % 2:
        x = [i for i in range(h + 1) if i % 2]
    # case: even
    else:
        x = [i for i in range(h + 1) if not i % 2]
    return partition[x]

# TODO: test this.
def hops_to_coverage_points_in_bounds(parentBounds,bounds,h):

    k = bounds.shape[0]
    cp = hops_to_coverage_points__standard(k,h)

    if is_proper_bounds_vector(bounds):
        s = [point_on_bounds_by_ratio_vector(bounds,c) for c in cp]
    else:
        s = [point_on_improper_bounds_by_ratio_vector(\
            parentBounds,bounds,c) for c in cp]
    return np.array(s)

'''

'''
def coverage_ratio_to_distance(boundsEDistance, h,cr):
    assert h >= 1.0, "invalid h"
    assert cr >= 0.0 and cr <= 1.0, "invalid cr"

    total = boundsEDistance / h
    return total * cr




#################################### end : ostracio && deletira


'''
'''
def RCHF__point_in_bounds_subvector_selector(b):
    def qf(xi):
        return operator.le(xi[1],b[xi[0],1]) and operator.ge(xi[1],b[xi[0],0])

    q2 = subvector_selector(qf,inputType = 2,outputType = 1)
    kwargs = ['nr', q2]
    rc = RChainHead()
    rc.add_node_at(kwargs)
    return rc.apply

from .poly_struct import *

"""
constructs an RCH function, 2+ nodes, node 1 outputs a float value from .arg<v>, last node outputs a bool|float
"""
def RCHF__ISPoly(x:'float',largs):
    rc = RChainHead()

    isp = ISPoly(x)

    def qf(v):
        return isp.apply(v)

    kwargs = ['nr',qf]
    rc.add_node_at(kwargs)

    for a in largs:
        rc.add_node_at(a)
    return rc.apply

"""
outputs the func for in bounds
"""
def RCHF___in_bounds(bounds0):
    kwargs = ['nr', lambda_pointinbounds, bounds0]
    # f : v -> v::bool
    rc = RChainHead()
    rc.add_node_at(kwargs)

    # f : filter out True | False
    subvectorSelector = boolies
    ss = subvector_selector(subvectorSelector,2)
    kwargs = ['nr',ss]
    rc.add_node_at(kwargs)

    # f : get indices
    ss = column_selector([0],True)
    kwargs = ['nr',ss]
    rc.add_node_at(kwargs)

    # f : apply indices on reference
    kwargs = ['nr',(vector_index_inverse_selector,[0])]
    rc.add_node_at(kwargs)

    return rc.apply

"""
pass string is boolean expression and the chains is pass vector<distances> -> <bool> --> pass-string -> bool
"""
def RCHF__point_distance_to_references_dec(r,ed0,passString):
    return -1

def ffilter(v,f):
    t,l = [],[]
    for v_ in v:
        if f(v_): t.append(v_)
        else: l.append(v_)

    return t,l

# make an rch by the following:
'''
using reference rf,
odd_multiply
even

[0] multiplier of even indices
[1] multiplier of odd indices

recent memory
+ - => -
- - => -
- + => +
+ + => +

past memory
+ - => +
- - => -
- + => -
+ + => +
'''
def rpmem_func(rf,rOp):
    rfo0,rfo1 = ffilter(rf,lambda i: i % 2)
    r1 = np.product(rfo1) if rOp else np.product(rfo0)

    def p(v):
        v0,v1 = ffilter(v,lambda i: i % 2)
        r2 = np.product(v1) if rOp else np.product(v0)
        return int(r2) % 2

    # try swapping them

    return p

"""
is permutation map valid?
"""
def is_valid_pm(pm):
    assert pm.shape[1] == 2, "invalid pm shape"
    tf1 = len(np.unique(pm[:,0])) == pm.shape[0]
    tf2 = len(np.unique(pm[:,1])) == pm.shape[0]
    return tf1 and tf2

"""
is permutation map proper?

"proper" := valid and all valids in range [0,n-1]
"""
def is_proper_pm(pm):
    s = is_valid_pm(pm)
    if not s: return s
    s1 = min(pm[:,0]) == 0 and max(pm[:,0])  == pm.shape[0]
    s2 = min(pm[:,1]) == 0 and max(pm[:,1])  == pm.shape[0]
    return s1 and s2

# a version of rp mem that uses a permutation map
# TODO
def rpmem_func__pm(rfs,pm):
    """
    the original func<rpmem_func> operates on the binary choice
    """
    assert is_proper_pm(pm), "[0] invalid permutation map"
    assert pm.shape[0] == len(rfs) + 1,"[1] invalid permutation map"
    return -1

def is_valid_subset_sequence(s,n):
    q = []
    for s_ in s:
        q.extend(list(s))

    tf0 = len(q)
    q = np.unique(q)
    tf1 = len(q)
    if tf0 != tf1: return False

    m0,m1 = min(q),max(q)
    return m0 == 0 and m1 == n - 1

###### END: functions used for relevance zoom


###### a sample RCH w/ update functionality



def sample_rch_1_with_update(parentBounds, bounds, h, coverageRatio):
    '''
vector -> ed vector -> (ed in bounds):bool
    '''
    def dm(rp,v):
        return np.array([euclidean_point_distance(v,rp_) for rp_ in rp])

    def cf(ds,dt_):
        return np.any(ds <= dt_)

    def update_dt_function(parentBounds,bounds,h,coverageRatio):
        return (euclidean_point_distance_of_bounds(parentBounds,bounds) / h)\
                    * coverageRatio

    rch = RChainHead()
    # add the node
    rf = hops_to_coverage_points_in_bounds(parentBounds,bounds,h)
    dm = dm
    cf = cf
    ed = euclidean_point_distance_of_bounds(parentBounds,bounds)
    dt = coverage_ratio_to_distance(ed,9,coverageRatio)

    kwargs = ['r',rf,dm,cf,dt]
    rch.add_node_at(kwargs)

    # add update functionality
    rch.s[0].updateFunc = {'rf': hops_to_coverage_points_in_bounds,\
            'dt': update_dt_function}
    rch.s[0].updatePath = {'rf': [0,1,2],'dt':[0,1,2,3]}
    rch.updatePath = {0: [0,1,2,3]}
    return rch

def sample_rch_2_with_update(parentBounds, bounds):
    """
    activation range is 0-20 percentile and 80-100 percentile
    """

    def activation_range(parentBounds,bounds):
        v = np.ones((parentBounds.shape[0],)) * 0.2
        b1s = np.copy(bounds[:,0])
        b1e = point_on_improper_bounds_by_ratio_vector(parentBounds,bounds,v)

        v = np.ones((parentBounds.shape[0],)) * 0.8
        b2s = point_on_improper_bounds_by_ratio_vector(parentBounds,bounds,v)
        b2e = np.copy(bounds[:,1])

        B1 = np.vstack((b1s,b1e)).T
        B2 = np.vstack((b2s,b2e)).T

        return B1,B2

    def update_dt_function(parentBounds,bounds):
        b1,b2 = activation_range(parentBounds,bounds)
        return (np.copy(parentBounds),b1,b2)

    """
    r := parentBounds,b1,b2
    """
    def cf(p,r):
        if point_in_improper_bounds(r[0],r[1],p):
            return True
        if point_in_improper_bounds(r[0],r[2],p):
            return True
        return False

    rch = RChainHead()
    dt = update_dt_function(parentBounds,bounds)
    kwargs = ['nr',cf,dt]
    rch.add_node_at(kwargs)

    # add update functionality
    rch.s[0].updateFunc = {'dt': update_dt_function}
    rch.s[0].updatePath = {'dt':[0,1]}
    rch.updatePath = {0: [0,1]}

    return rch
