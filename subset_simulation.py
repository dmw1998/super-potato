# Subset simulation

from kl_expan import kl_expan_2
from IoQ_and_c_l import *
from sampling_MC import *
import numpy as np

def subset_simulation(p0, N, M, gamma = 0.8, u_max = 0.535, n_grid = 4, L = 4):
    # input:
    # p0: probability of failure
    # N: number of samples
    # M: length of theta
    # gamma: correlation parameter
    # u_max: critical value
    # n_grid: number of grid points
    # L: number of levels
    
    # output:
    # p_f: final probability of failure
    
    # Generate initial samples
    theta_ls = [np.random.randn(M) for _ in range(N)]
    G = [u_max - IoQ(kl_expan_2(theta), n_grid) for theta in theta_ls]
        
    # Determine the threshold value c_l
    G, theta_ls, c_l = failure_level(G, theta_ls, N, 0, p0 = p0, L = L)
    print('c_0 = ', c_l)
    
    if c_l < 0:
        return len(G) / N
    else:
        p_f = p0

    for l in range(1,L):
        G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma)
        G, theta_ls, c_l = failure_level(G, theta_ls, N, l, p0 = p0, L = L)
        
        print('c_l =', c_l)
        if c_l < 0:
            # When we reach the finest level
            P_L = len(G) / N
            p_f *= P_L
            break
        else:
            p_f *= p0
        
    # Compute the final probability of failure
    G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma)
    G, theta_ls, c_l = failure_level(G, theta_ls, N, L, p0 = p0, L = L)
    print('c_L =', c_l)
    P_L = len(G) / N
    if P_L == 0:
        print('P_L = 0')
    else:
        print('P_L =', P_L)
        p_f *= P_L
        
    return p_f

if __name__ == '__main__':
    p0 = 0.1
    N = 1000
    M = 150
    p_f = subset_simulation(p0, N, M)
    print('Final probability of failure:', p_f)