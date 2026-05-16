from morebs2.v2f_solver import * 
from morebs2.numerical_generator import prg__single_to_nvec
import unittest

### lone file test 
"""
py -m morebs2.tests.v2f_solver_test
"""
###
class Vector2FloatSolverTypeS1Class(unittest.TestCase):

    def test__Vector2FloatSolverTypeS1__solve__case_1(self):

        V = np.array([5,0,6,7,-5,-13]) 
        prg = prg__LCG(45,-677,899,1900.05)

        f = 1200 
        f2 = -2456 

        # subcase 1 
        vs = Vector2FloatSolverTypeS1(V,f,prg)
        vs.solve() 
        assert round(abs(vs.output() - f),4) == 0 
        print("solved {} --> \n\t{} *\n\t{}".format(f,V,np.round(vs.W,5))) 

        # subcase 2 
        vs2 = Vector2FloatSolverTypeS1(V,f2,prg)
        vs2.solve() 
        assert round(abs(vs2.output() - f2),4) == 0 
        print("solved {} --> \n\t{} *\n\t{}".format(f2,V,np.round(vs2.W,5))) 

        # subcases 3-6
        q = [5,4,10,8] 
        for q_ in q: 
            V2 = prg__single_to_nvec(prg,q_)()
            f_ = (prg() - prg()) * (prg() - prg())
            vs3 = Vector2FloatSolverTypeS1(V2,f_,prg)
            vs3.solve() 
            assert round(abs(vs3.output() - f_),4) == 0 
            print("solved {} --> \n\t{} *\n\t{}".format(f_,V2,np.round(vs3.W,5)))  

    def test__Vector2FloatSolverTypeS1__solve__case_2(self):

        V = np.array([5,0,0,7,0,0])  
        prg = prg__LCG(45,-677,899,1900.05)

        f = 44567 
        f2 = -845611 

        vs = Vector2FloatSolverTypeS1(V,f,prg)
        vs.solve() 
        assert round(abs(vs.output() - f),4) == 0 

        vs2 = Vector2FloatSolverTypeS1(V,f2,prg)
        vs2.solve() 
        assert round(abs(vs2.output() - f2),4) == 0 

        q = [5,4,10,8] 
        for q_ in q: 
            V2 = prg__single_to_nvec(prg,q_)()
            f_ = (prg() - prg()) * (prg() - prg())
            vs3 = Vector2FloatSolverTypeS1(V2,f_,prg)
            vs3.solve() 
            assert round(abs(vs3.output() - f_),4) == 0 

class Vector2FloatSolverTypeRXClass(unittest.TestCase): 

    def test__Vector2FloatSolverTypeRX__solve__case_1(self):

        V3 = np.array([-4,12,32,0,5]) 

        B = np.zeros((5,2)) 
        B[:,0] = -50 
        B[:,1] = 50 

        prg3 = prg__LCG(245.6,-1313.2,245.76,4019.4) 

        # solve min  
        f = -2250 
        S3 = Vector2FloatSolverTypeRX(V3,B,f,prg3) 
        S3.solve() 

        assert round(S3.output()-f,5) == 0. 

        # solve max 
        f2 = 2250-0.05 
        S4 = Vector2FloatSolverTypeRX(V3,B,f2,prg3) 
        S4.solve()
        q = S4.output() 
        assert round(q-f2,5) < 10**-4, "got {}".format(round(q - f2,5)) 

        # solve in-between 
        f3 = 995 
        S5 = Vector2FloatSolverTypeRX(V3,B,f3,prg3) 
        S5.solve()
        q = S5.output() 
        assert round(q-f3,5) < 10**-4, "got {}".format(round(q - f3,5)) 

        # solve 0 
        f4 = 0  
        S6 = Vector2FloatSolverTypeRX(V3,B,f4,prg3) 
        S6.solve()
        q = S6.output() 
        assert round(q-f4,5) < 10**-4, "got {}".format(round(q - f4,5)) 

if __name__ == '__main__':
    unittest.main()