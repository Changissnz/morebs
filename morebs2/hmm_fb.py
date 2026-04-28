from .measures import * 

def check_HMM_table_(T,wanted_keys): 
    for v in T.values(): 
        kv = set(v.keys()) 
        assert kv == wanted_keys 
        for v2 in v.values(): 
            assert 0. <= v2 <= 1. 

def check_HMM_tables(T,B): 
    assert type(T) == dict == type(B) 

    K = set(T.keys()) 
    check_HMM_table_(T,K) 
    check_HMM_table_(B,K)

# NOTE: caution. not fully verified to be correct. 
# BUG (possible): order-of-operations may be off. 
class ForwardBackward: 

    def __init__(self,T,B): 
        check_HMM_tables(T,B)  
        self.T = T 
        self.B = B 

        self.obs_seq = None 

        self.hidden_states = sorted(self.T.keys())

        self.obsstate2mat_map = None 
        self.starting_pr = None 

        self.pr_forward = None 
        self.pr_backward = None 

        self.preproc() 

    def preproc(self): 
        self.T_map_to_table()
        self.observations_to_matrix_map()

    def load_new_obs_seq(self,O,starting_pr=None): 
        assert type(O) == list 
        for o in O: assert o in self.B 
        self.obs_seq = O 

        l = len(self.hidden_states)
        if type(starting_pr) == type(None): 
            self.starting_pr = np.ones((l,)) * 1 / l 
        else: 
            assert is_vector(starting_pr)
            assert len(starting_pr) == l 
            self.starting_pr = starting_pr
        
        self.pr_forward = [] 
        self.pr_backward = [] 
        self.pr_smoothed = [] 

    def add_one_observation(self,o): 
        assert o in self.B 
        self.obs_seq.append(O)

    """
    main method 
    """
    def run(self,O,starting_pr=None):
        self.load_new_obs_seq(O,starting_pr)

        self.forward() 
        self.backward()
        self.smoothing() 
        return

    def forward(self): 
        l = len(self.obs_seq) 

        for i in range(l): 
            self.forward_at(i,record_value=True)
        return

    def forward_at(self,index,record_value:bool=False): 
        assert 0 <= index < len(self.obs_seq)
        o = self.obs_seq[index] 
        O = self.obsstate2mat_map[o] 


        if index == 0: 
            P = self.starting_pr
        else: 
            P = self.pr_forward[index-1] 

        q = np.dot(self.T_mat.T,P) 
        X = normalize_vector(np.dot(O,q)) 

        if record_value:
            self.pr_forward.append(X) 
        return X 

    def backward(self):
        l = len(self.obs_seq) 

        for i in range(l-1,-1,-1): 
            self.backward_at(i,record_value=True) 
        return     

    def backward_at(self,index,record_value:bool=False): 
        assert 0 <= index < len(self.obs_seq)
        o = self.obs_seq[index] 
        O = self.obsstate2mat_map[o] 

        if index == len(self.obs_seq) - 1:  
            P = np.ones((len(self.hidden_states),))
        else: 
            P = self.pr_backward[0]  

        q = np.dot(O,P) 
        X = normalize_vector(np.dot(self.T_mat,q)) 

        if record_value:
            self.pr_backward.insert(0,X) 
        return 

    def smoothing(self): 
        for i in range(len(self.obs_seq) + 1): 
            self.smooth_at(i,record_value=True)

        return

    def smooth_at(self,index,record_value:bool=False): 

        if index == 0: 
            P = self.starting_pr
        else: 
            P = self.pr_forward[index-1] 

        if index == len(self.obs_seq): 
            P2 = np.ones((len(self.hidden_states),)) 
        else: 
            P2 = self.pr_backward[index]

        pr = P * P2 
        pr = normalize_vector(pr) 

        if record_value:
            self.pr_smoothed.append(pr) 
        return pr 

    #--------------------------- formatting methods, from dict to matrix. 

    def T_map_to_table(self): 
        l = len(self.T)
        self.T_mat = np.zeros((l,l)) 

        for (i,h) in enumerate(self.hidden_states): 
            for (j,h2) in enumerate(self.hidden_states): 
                self.T_mat[i,j] = self.T[h][h2] 
    
    def observations_to_matrix_map(self): 
        self.obsstate2mat_map = dict()
        for o in self.B.keys(): 
            M = self.obs_to_mat(o) 
            self.obsstate2mat_map[o] = M 
        return 

    def obs_to_mat(self,obs): 
        assert obs in self.B 

        l = len(self.hidden_states)
        M = np.zeros((l,l)) 
        
        for (i,h) in enumerate(self.hidden_states): 
            M[i,i] = self.B[obs][h] 
        return M  