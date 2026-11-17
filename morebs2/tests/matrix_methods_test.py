from morebs2.matrix_methods import *
import unittest

def stringizable_vars__case_FOUR():
    W = 45 
    W1 = ["qsdfsa",56,733] 
    W2 = {"TT",256,733, "YY"} 
    W3 = np.array([456,523,390,4621]) 
    return W,W1,W2,W3

'''
py -m morebs2.tests.matrix_methods_test    
'''
class TestStrConversionFunctions(unittest.TestCase):

    def test_stringize_common_var__case_1(self):
        W,W1,W2,W3 = stringizable_vars__case_FOUR()

        q0 = stringize_common_var(W)
        q1 = stringize_common_var(W1)
        q2 = stringize_common_var(W2)
        q3 = stringize_common_var(W3)

        assert q0 == '45'
        assert q1 == 'qsdfsa,56,733'
        assert q2 == '256,733,TT,YY'
        assert q3 == '456,523,390,4621'
        
    def test_destringize_common_var__case_1(self):
        W,W1,W2,W3 = stringizable_vars__case_FOUR()

        q0 = stringize_common_var(W)
        q1 = stringize_common_var(W1)
        q2 = stringize_common_var(W2)
        q3 = stringize_common_var(W3)

        W_0 = destringize_common_var(q0,float,output_set=True)
        W_1 = destringize_common_var(q0,float,output_set=False) 

        assert W_0 == {45.0}
        assert W_1 == 45.0
        for q in W_0: assert type(q) == float 
        assert type(W_1) == float 


        W1_0 = destringize_common_var(q1,str,output_set=True)
        W1_1 = destringize_common_var(q1,str,output_set=False) 

        assert W1_0 == {'56', '733', 'qsdfsa'}
        assert W1_1 == ['qsdfsa', '56', '733']


        W2_0 = destringize_common_var(q2,str,output_set=True)
        W2_1 = destringize_common_var(q2,str,output_set=False) 

        assert W2_0 == {'YY', 'TT', '256', '733'}
        assert W2_1 == ['256', '733', 'TT', 'YY']


        W3_0 = destringize_common_var(q3,float,output_set=True)
        W3_1 = destringize_common_var(q3,float,output_set=False) 

        assert W3_0 == {456.0, 523.0, 4621.0, 390.0}
        for x in W3_0: assert type(x) == float 
        assert W3_1 == [456.0, 523.0, 390.0, 4621.0]
        for x in W3_1: assert type(x) == float 

if __name__ == '__main__':
    unittest.main()
