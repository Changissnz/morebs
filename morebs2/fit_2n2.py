
'''
2 and 2 fit: various fit-formulae between two points
            in a 2-d space. 
'''
from morebs2.distributions import *

class Fit22:

    def __init__(self,ps,direction=[0,1]):
        assert ps.shape == (2,2)
        assert len(direction) == 2 and set(direction) == {0,1}, "invalid direction"
        self.ps = ps
        self.direction = direction
        assert self.ps[self.direction[1],1] > self.ps[self.direction[0],1], "invalid y's"
        self.f = None 


class LogFit22(Fit22):
    '''
    structure fits a 
    '''

    def __init__(self,ps,direction=[0,1]):
        super().__init__(ps,direction)
        self.f = self.fit() 
        self.g = self.yfit()

    def fit(self):

        def f(x):
            # ratio of x on x-span
            r1 = abs(x - self.ps[self.direction[0],0]) / abs(self.ps[1,0] - self.ps[0,0])
            real = log(r1 * 9. + 1.) / log(10.)
            return self.ps[self.direction[0],1] + real * abs(self.ps[0,1] - self.ps[1,1])    
        return f

    def yfit(self):

        def g(y):

            p = log(10) * (y - self.ps[self.direction[0],1]) / (abs(self.ps[self.direction[1],1] - self.ps[self.direction[0],1]))
            ep = (e ** p - 1) / 9
            r = ep * abs(self.ps[1,0] - self.ps[0,0])
            x1 = r - self.ps[self.direction[0],0]
            x2 = r + self.ps[self.direction[0],0]
            if round(x1,5) >= round(min(self.ps[:,0]),5) and round(x1,5) <= round(max(self.ps[:,0]),5):
                return x1

            if round(x2,5) >= round(min(self.ps[:,0]),5) and round(x2,5) <= round(max(self.ps[:,0]),5):
                return x2
            raise ValueError("invalid y-input") 

        return g

# case 1
'''
lf22 = LogFit22(np.array([[30.,12.],[60.,80.]]))
function_to_2dplot(generate_2d_data_from_function,\
    lf22.f,[30.,60.],0.1,())
'''

# case 2
'''
lf22 = LogFit22(np.array([[0.,0.],[1.,1.]]))
function_to_2dplot(generate_2d_data_from_function,\
    lf22.f,[0.,1.],0.1,())
'''

# case 3
'''
lf22 = LogFit22(np.array([[60.,12.],[30.,80.]]))
function_to_2dplot(generate_2d_data_from_function,\
    lf22.f,[30.,60.],0.1,())
'''

# case 4
"""
lf22 = LogFit22(np.array([[60.,80.],[30.,12.]]),direction=[1,0])
function_to_2dplot(generate_2d_data_from_function,\
    lf22.f,[30.,60.],0.1,())
"""

class Exp2Fit22(Fit22):
    '''
    structure fits a 
    '''

    def __init__(self,ps,direction=[0,1]):
        super().__init__(ps,direction)
        self.f = self.fit() 
        self.g = self.yfit() 

    def fit(self):

        def f(x):
            # ratio of x on x-span
            r1 = abs(x - self.ps[self.direction[0],0]) / abs(self.ps[1,0] - self.ps[0,0])
            real = r1 ** 2
            return self.ps[self.direction[0],1] + real * abs(self.ps[0,1] - self.ps[1,1])    
        return f

    def yfit(self):

        def g(y):
            m1 = (y - self.ps[self.direction[0],1]) / abs(self.ps[self.direction[0],1] - self.ps[self.direction[1],1])
            m2 = (self.ps[self.direction[0],0] - self.ps[self.direction[1],0]) ** 2
            print("M1 M2 ",m1, m2)
            m = sqrt(m1 * m2)

            x1 = m + self.ps[self.direction[0],0] 
            x2 = m - self.ps[self.direction[0],0]

            if x1 >= min(self.ps[:,0]) and x1 <= max(self.ps[:,0]):
                return x1

            if x2 >= min(self.ps[:,0]) and x2 <= max(self.ps[:,0]):
                return x2
            raise ValueError("invalid y-input") 
            
        return g

# case 1
'''
lf22 = Exp2Fit22(np.array([[60.,80.],[30.,12.]]),direction=[1,0])
function_to_2dplot(generate_2d_data_from_function,\
    lf22.f,[30.,60.],0.1,())
'''
#########


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
            return self.fs.y_given_x(x)
        return self.fs.f(x)

    def __str__(self):
        l = None
        if type(self.fs) == Line: l = "line"
        elif type(self.fs) == LogFit22: l = "logfit22"
        else: l = "exp2fit22"
        p = self.get_point()

        s = "* struct: {} \n".format(l)
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

    def in_point_range(self,p):
        pr = self.point_range()
        # case: in y-range?
        if self.ad in {'l','r'}:
            return p[1] >= pr[1,0] and p[1] <= pr[1,1]
        # case: in x-range?
        else:
            return p[0] >= pr[0,0] and p[0] <= pr[0,1]

    def is_ap(self,p):
        '''
        is activation point?
        '''
        print("PR: ",self.point_range(),"\t",p)

        if self.ad in {'t','b'}:
            p2 = self.y_given_x(p[0])
            if self.ad == 't':
                return p2 <= p[1]
            return p2 >= p[1]
        else:
            p2 = self.x_given_y(p[1])
            if self.ad == 'l':
                return p2 >= p[0]
            return p2 <= p[0]

    def modulate_fit(self):
        if type(self.fs) == Line:
            return

        if type(self.fs) == LogFit22:
            c = LogFit22(deepcopy(self.fs.ps),self.fs.direction)
        else:
            c = Exp2Fit22(deepcopy(self.fs.ps),self.fs.direction)
        self.fs = c

