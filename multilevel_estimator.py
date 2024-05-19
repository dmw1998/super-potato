# Find the probability of failure with a multilevel estimator
from fenics import *
import numpy as np
from kl_expan import kl_expan_2
from IoQ_and_c_l import *
from sampling_MC import *

def mle(p0, M, N, L_b, u_max = 0.535, n_grid = 4, L = 6):
    # input:
    # p0: probability of failure
    # M: length of theta
    # N: number of samples
    # L_b: burn-in length
    # u_max: critical value
    # n_grid: number of grid points
    # L: number of levels, L should larger than 3
    
    # output:
    # p_f: final probability of failure
    
    # Generate initial samples
    theta_ls = [np.random.randn(M) for _ in range(N)]
    G = [u_max - IoQ(kl_expan_2(theta), n_grid) for theta in theta_ls]
    
    # Determine the threshold value c_l
    G, theta_ls, c_l = failure_level(G, theta_ls, N, 0, p0 = p0, L = L)
    print('c_0 = ', c_l)
    print('G =', len(G))
    
    # Stop if the threshold value is negative
    if c_l < 0:
        return len(G) / N
    else:
        denominator = 1
        
    # l < 3, set L_b = 0
    c_l_1 = c_l
    G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma = 0.8)
    G, theta_ls, c_l = failure_level(G, theta_ls, N, 1, p0 = p0, L = L)
    print('c_1 = ', c_l)
    print('G =', len(G))
    G, theta_ls = MCMC_sampling_burn_in(0, N, G, theta_ls, u_max, c_l, gamma = 0.8)
    
    G_in_l_1 = [g for g in G if g <= c_l_1]
    print('G_in_l_1 =', len(G_in_l_1))
    denominator *= len(G_in_l_1) / N
    
    # 2 < l < L, set L_b = L_b
    for l in range(2, L-1):
        c_l_1 = c_l
        G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma = 0.8)
        G, theta_ls, c_l = failure_level(G, theta_ls, N, l, p0 = p0, L = L)
        print('c_', l, ' = ', c_l)
        print('G =', len(G))
        
        if c_l < 0:
            print('denominator =', denominator)
            return p0 ** l * len(G) / N / denominator
        
        G, theta_ls = MCMC_sampling_burn_in(L_b, N, G, theta_ls, u_max, c_l, gamma = 0.8)
        
        G_in_l_1 = [g for g in G if g <= c_l_1]
        denominator *= len(G_in_l_1) / N
        
    # l = L (in program, l = L - 1)
    G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma = 0.8)
    G, theta_ls, c_l = failure_level(G, theta_ls, N, L, p0 = p0, L = L)
    print('c_', L, ' = ', c_l)
    P_L_L_1 = len(G) / N
    
    p_f = p0 ** L * P_L_L_1 / denominator
    
    return p_f

if __name__ == '__main__':
    p0 = 0.25
    M = 150
    N = 100
    L_b = 10
    print(mle(p0, M, N, L_b))