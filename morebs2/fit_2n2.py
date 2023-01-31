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

    def fit(self):

        def f(x):
            # ratio of x on x-span
            r1 = abs(x - self.ps[self.direction[0],0]) / abs(self.ps[1,0] - self.ps[0,0])
            real = log(r1 * 9. + 1.) / log(10.)
            return self.ps[self.direction[0],1] + real * abs(self.ps[0,1] - self.ps[1,1])    
        return f

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

lf22 = LogFit22(np.array([[60.,80.],[30.,12.]]),direction=[1,0])
function_to_2dplot(generate_2d_data_from_function,\
    lf22.f,[30.,60.],0.1,())


class Exp2Fit22(Fit22):
    '''
    structure fits a 
    '''

    def __init__(self,ps,direction=[0,1]):
        super().__init__(ps,direction)
        self.f = self.fit() 

    def fit(self):

        def f(x):
            # ratio of x on x-span
            r1 = abs(x - self.ps[self.direction[0],0]) / abs(self.ps[1,0] - self.ps[0,0])
            real = r1 ** 2
            return self.ps[self.direction[0],1] + real * abs(self.ps[0,1] - self.ps[1,1])    
        return f

# case 1
'''
lf22 = Exp2Fit22(np.array([[60.,80.],[30.,12.]]),direction=[1,0])
function_to_2dplot(generate_2d_data_from_function,\
    lf22.f,[30.,60.],0.1,())
'''