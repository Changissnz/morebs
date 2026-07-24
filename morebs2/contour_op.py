from .matrix_methods import * 
from types import MethodType,FunctionType
from .numerical_generator import prg_decimal#,modulo_in_range
import random 

"""
checks if sequence `cpoints` represents contour points, as defined in 
the <morebs2> program as 
    (degree,x,y). 
If `cpoints` are contour points, returns points ordered by degree. 
Otherwise, outputs null. 
"""
def are_contour_points(cpoints,require_0:bool):
    if not type(cpoints) == np.ndarray: return None 

    if not (len(cpoints.shape) == 2 and cpoints.shape[1] == 3): 
        return None 

    cpoints = np.array(sorted(cpoints,key=lambda x:x[0])) 

    if require_0: 
        if cpoints[0,0] != 0.: return None 
    return cpoints

class ContourInterpolation: 

    def __init__(self,cpoints,prg=None,add_noise:bool=True,require_0:bool=True): 
        cpoints = are_contour_points(cpoints,require_0) 
        assert type(cpoints) != type(None) 
        self.cpoints = cpoints 
        
        assert type(prg) in {MethodType,FunctionType,type(None)}
        assert type(add_noise) == bool 

        if type(prg) == type(None): 
            def f(): 
                r0 = random.uniform(0.,2048.) 
                return round(r0,5) 
            prg = f 
        self.prg = prg 
        self.add_noise = add_noise 

        return 

    def generate_points(self,points_bt_each_pair=5): 
        lx = [] 
        for i in range(len(self.cpoints)): 
            lx_ = self.generate_points_bt_cpoint_pair(i,points_bt_each_pair) 
            lx.extend(lx_)
        return np.round(lx,5) 

    def generate_points_bt_cpoint_pair(self,start_index,points_bt_pair): 
        end_index = (start_index + 1) % len(self.cpoints) 
        deg0,deg1 = self.point_indexpair_to_degrees(start_index,end_index) 
        dprt = n_partition_for_range((deg0,deg1),points_bt_pair + 1)
        q = np.array([self.point_at_degree_(d,start_index) for d in dprt]) 
        if self.add_noise: 
            return self.add_noise_to_sequence(q,(self.cpoints[start_index,1:],\
                self.cpoints[end_index,1:]))
        return q 

    def add_noise_to_sequence(self,point_sequence,ref_point_pair): 
        if len(point_sequence) <= 2: return point_sequence 

        D = euclidean_point_distance(ref_point_pair[0],ref_point_pair[1]) 
        l = (D / 2) / (len(point_sequence) - 1) 

        for (i,p) in enumerate(point_sequence): 
            if i == 0: continue 
            if i == len(point_sequence) - 1: continue 

            s0 = 1 if prg_decimal(self.prg,[0,1.]) >= 0.5 else -1 
            s1 = 1 if prg_decimal(self.prg,[0,1.]) >= 0.5 else -1 

            xdelta = s0 * (self.prg() % l)
            ydelta = s1 * (self.prg() % l) 
            point_sequence[i] = (xdelta + point_sequence[i,0],\
                    ydelta + point_sequence[i,1])
        return point_sequence 

    def point_at_degree(self,d,cpoint_start_index=None): 
        i0 = cpoint_start_index if type(cpoint_start_index) != type(None) \
            else self.degree_to_cpoint_index(d) 

        i1 = (i0 + 1) % len(self.cpoints)
        p0,p1 = self.cpoints[i0],self.cpoints[i1] 
        x,y = self.point_at_degree_(d,cpoint_start_index)  

        # case: either X or Y is equal, go with line interpolation 
        if abs(round(p0[0] - p1[0],5)) == 0. or abs(round(p0[1] - p1[1],5)) == 0.: 
            return (x,y)
 
        p0_,p1_ = p0[1:],p1[1:]
        F = LogFit22(np.array([p0_,p1_])) 

        pa0 = F.f(x) 
        pa1 = F.g(y) 

        return tuple((np.array([x,pa0]) + np.array([pa1,y])) / 2.0) 
        
    # NOTE: code somewhat duplicate of methods in class<Line> 
    def point_at_degree_(self,d,cpoint_start_index=None): 
        
        d = round(d % 360.,5) 
        i0 = cpoint_start_index if type(cpoint_start_index) != type(None) \
            else self.degree_to_cpoint_index(d) 
    
        i1 = (i0 + 1) % len(self.cpoints)
        deg0,deg1 = self.point_indexpair_to_degrees(i0,i1)

        if not deg0 <= d <= deg1:
            assert d == round(deg1 % 360,5) 
            r0 = 1.
        else: 
            r0 = (d - deg0) / (deg1 - deg0) 

        p0,p1 = self.cpoints[i0],self.cpoints[i1] 

        xdiff = p1[1] - p0[1]
        ydiff = p1[2] - p0[2] 

        return tuple(np.round((p0[1] + (xdiff * r0),\
            p0[2] + (ydiff * r0)),5)) 

    def degree_to_cpoint_index(self,d): 
        d = d % 360 

        l = len(self.cpoints)
        for i in range(l):
            i1 = (i + 1) % l 
            
            deg0,deg1 = self.point_indexpair_to_degrees(i,i1) 
            if deg0 <= d <= deg1: 
                return i 
        assert False 

    def point_indexpair_to_degrees(self,i0,i1): 
        if i1 == 0: 
            return self.cpoints[i0,0],360. 
        return self.cpoints[i0,0],self.cpoints[i1,0]  

class ContourPointMod: 

    def __init__(self,center,cpoints,radius_range):  
        cpoints = are_contour_points(cpoints,False) 
        assert type(cpoints) != type(None)
        self.cpoints = cpoints 
        return 

    def alter_point(self,point_index): 
        return -1 

    def add_point_after(self,point_index):  
        return -1 
