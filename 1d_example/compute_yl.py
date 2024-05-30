# To compute the sequence of failure thresbolds y_l

import numpy as np

def y_l(L, gamma):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    
    # output:
    # y: a sequence of the threshold values
    
    y = np.zeros(L)
    for i in range(L-1):
        l = L - 2 - i
        y[l] = (gamma ** l + gamma ** (l+1)) + y[l+1]
        
    return y