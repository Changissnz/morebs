"""
file for simulation of recursive processes
"""
from types import MethodType,FunctionType


"""
Simulates a recursive method. 

Is comprised of n conditional statements, n > 0, and a possible NOT conditional. 
Every conditional has 4 elements: 
[0] boolean function : accept or reject parameters.
    - null for NOT conditional.  
[1] parameter delta function: modifies parameters. 
    - optional; can be null. 
[2] output function: maps parameters to an output value. 
    - optional; can be null. 
[3] termination function: outputs boolean for termination of recursive process on `parameters.

The `pre_delta_function` P is optional. 

----------------------------------------------------------------------------------------------

Process is first set with starting `parameters`, via method<load_parameters>. 

For every recursive call (method<__next__>): 
- If P is not null, apply 
    `parameters` = P(`parameters`). 
- Iterate through the conditionals c_i in C: 
    if c_i[0](`parameters`): 
        Process `parameters` using that conditional. 
        Quit call.
- If no conditionals apply, then apply the NOT conditional. 
"""
class SimulatedRecursionNode:

    def __init__(self,pre_delta_function,conditionals,not_conditional): 
        assert type(pre_delta_function) in {type(None),MethodType,FunctionType}
        self.pre_delta_function = pre_delta_function

        assert len(conditionals) > 0 
        for x in conditionals: 
            self.assert_proper_cond(x,False) 
        self.conditionals = conditionals

        self.assert_proper_cond(not_conditional,True)
        self.not_conditional = not_conditional
        self.parameters = None
        self.output_seq = [] 

        self.terminated = False 
        self.c = 0 
        return

    def set_pre_delta_function(self,p): 
        assert type(p) in {type(None),MethodType,FunctionType}
        self.pre_delta_function = p 

    def add_conditional(self,c): 
        self.assert_proper_cond(c,False) 
        self.conditionals.append(c) 
        return 

    def replace_conditional(self,c,index): 
        assert 0 <= index < len(self.conditionals)
        self.assert_proper_cond(c,False) 
        self.conditionals[index] = c         
        return

    def set_NOT_conditional(self,c): 
        self.assert_proper_cond(c,True) 
        return

    def assert_proper_cond(self,x,is_not_cond:bool):
        if type(x) == type(None) and is_not_cond: return 

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
            stat_ = self.process_conditional_(self.not_conditional)
        return stat_ 