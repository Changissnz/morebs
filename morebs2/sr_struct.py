"""
file for simulation of recursive processes
"""
from types import MethodType,FunctionType

class SimulatedRecursionNode:

    def __init__(self,pre_delta_function,conditionals,not_conditional): 
        assert type(pre_delta_function) in {type(None),MethodType,FunctionType}
        self.pre_delta_function = pre_delta_function

        assert len(conditionals) > 0 
        for x in conditionals: 
            self.assert_proper_cond(x,False,False) 
        self.conditionals = conditionals

        self.assert_proper_cond(not_conditional,True,True)
        self.not_conditional = not_conditional
        self.parameters = None
        self.output_seq = [] 

        self.terminated = False 
        self.c = 0 
        return

    def assert_proper_cond(self,x,is_not_cond:bool,non_permitted:bool):
        if type(x) == type(None) and non_permitted: return 

        assert type(x) == tuple  
        assert len(x) == 4 
        
        # boolean condition 
        if not is_not_cond:
            assert type(x[0]) in {MethodType,FunctionType}
        else: 
            assert type(x[0]) == type(None) 

        # parameter delta 
        assert type(x[1]) in {MethodType,FunctionType,type(None)} 
        # output function 
        assert type(x[2]) in {MethodType,FunctionType,type(None)}
        # terminating condition 
        assert type(x[3]) in {MethodType,FunctionType} 
        return

    def load_parameters(self,parameter_seq): 
        assert type(parameter_seq) == list 
        self.parameters = parameter_seq 
        self.output_seq.clear() 
        self.terminated = False
        self.c = 0  
        return

    def process_conditional(self,i): 
        x = self.conditionals[i]
        return self.process_conditional_(x)

    def process_conditional_(self,x): 
        if type(x) == type(None): return False 

        q = x[0] 

        # NOT conditional 
        if type(q) == type(None): 
            self.one_proc(x) 
            return True 
        
        if q(*self.parameters): 
            self.one_proc(x) 
            return True 
            
        return False 

    def one_proc(self,x): 
        q1 = x[1] 
        if type(q1) != type(None): 
            self.parameters = q1(*self.parameters) 

        q2 = x[2]  
        if type(q2) != type(None): 
            output = q2(*self.parameters) 
            self.output_seq.append(output) 
        
        q3 = x[3] 
        self.terminated = q3(*self.parameters)  
        return 

    def __next__(self): 
        if self.terminated: return 
        self.c += 1

        if type(self.pre_delta_function) != type(None): 
            self.parameters = self.pre_delta_function(*self.parameters)

        stat_ = False 
        for i in range(len(self.conditionals)): 
            stat_ = self.process_conditional(i)
            if stat_: 
                break 

        if not stat_: 
            self.process_conditional_(self.not_conditional)