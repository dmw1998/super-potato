# Subset simulation

from kl_expan import kl_expan
from IoQ_and_c_l import *
from sampling_MC import *
import numpy as np

def subset_simulation(p0, N, M, gamma = 0.8, u_max = 0.535, n_grid = 400):
    # input:
    # p0: probability of failure
    # N: number of samples
    # M: length of theta
    # gamma: correlation parameter
    
    # output:
    # p_f: final probability of failure
    
    p_f = 1
    
    # Generate initial samples
    theta_ls = [np.random.rand(M)]
    u_1 = IoQ(kl_expan(theta_ls[0]), n_grid)
    G = [u_1 - u_max]
    
    for i in range(N):
        theta = gamma * theta_ls[i] + np.sqrt(1 - gamma**2) * np.random.randn(M)
        theta_ls.append(theta)
        g = IoQ(kl_expan(theta), n_grid) - u_max
        G.append(g)
        
    # Determine the threshold value c_l
    G, theta_ls, c_l = failure_level(G, theta_ls, N, p0)

    while True:
        if c_l < 0:
            print('c_l =', c_l)
            P_L = len(G) / N
            p_f *= P_L
            break
        else:
            print('c_l =', c_l)
            p_f *= p0
            
        G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma)
        G, theta_ls, c_l = failure_level(G, theta_ls, N, p0)
        
    return p_f

if __name__ == '__main__':
    p0 = 0.1
    N = 1000
    M = 150
    p_f = subset_simulation(p0, N, M)
    print('Final probability of failure:', p_f)