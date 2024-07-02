import numpy as np

class MCMC:
    def __init__(self, params: dict, data: np.ndarray) -> None:
        self.data = data
        #Do basic defintions
        
        
    def run():
        #Loop
        pass
    
    def corrections():
        #Do corrections
        pass
    
    def find_new_propostion():
        #Find new proposition
        pass
    
    def model_prior():
        #Model prior
        pass
    
    def model_post_proba():
        #Model posterior probability
        pass
    
    def accept_dismiss():
        #Accept or dismiss
        pass
    
    def save():
        #Save , also on the go
        pass
    
    #DECIMAL, BINARY AND VISUAL CONVERSION    
    def bin_to_dec(self, bin_array):
        '''Convert binary array to decimal number'''
        pows = 2 ** np.arange(len(bin_array))
        dec = np.sum(bin_array * pows)
        return dec
    
    def dec_to_bin(self, dec):
        '''Convert decimal number to binary array'''
        bin_array = np.zeros(self.max_poly_deg+1, dtype=np.int8)
        i = 0
        while dec > 0:
            bin_array[i] = dec % 2
            dec //= 2
            i += 1
        return bin_array