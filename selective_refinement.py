# Selective refinement for 1d diffusion equation

from fenics import *
from IoQ_and_c_l import *
from kl_expan import kl_expan_2
import numpy as np

def selective_refinement(l, thetas, u_max, c_l, n_grid = 1, tol_gamma = 0.1):
    # input:
    # l: level
    # thetas: list of theta
    # u_max: upper bound of IoQ
    # c_l: threshold value - critical value
    # tol_gamma: tolerance parameter
    
    # output:
    # G: updated G
    
    # Initialization
    j = 0
    G = u_max - IoQ(kl_expan_2(thetas), n_grid)
    
    while j < l and np.abs(G - c_l) < tol_gamma ** j:
        j += 1
        n_grid *= 2
        G = u_max - IoQ(kl_expan_2(thetas), n_grid)
        
    return G

def MCMC_sampling_sr(N, G, theta_ls, l, c_l, u_max, n_grid = 1, gamma = 0.8):
    # input:
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # l: level
    # c_l: threshold value
    # u_max: upper bound of the solution
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    M = len(theta_ls[0])
    i = 0
    while i < N - N0:
        theta = theta_ls[i]
        theta_new = gamma * theta + np.sqrt(1 - gamma**2) * np.random.randn(M)
        G_new = selective_refinement(l, theta_new, u_max, c_l, n_grid = n_grid)
        
        if G_new <= c_l:
            G.append(G_new)
            theta_ls.append(theta_new)
            # i += 1
        else:
            G.append(G[i])
            theta_ls.append(theta)
            
        i += 1
    
    return G, theta_ls


if __name__ == '__main__':
    u_max = 0.535
    l = 5
    N = 10
    M = 150
    c_l = 0.4
    
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        
        G = selective_refinement(l, thetas, u_max, c_l)
        
        print('G =', G)
    
    G, thetas = MCMC_sampling_sr(N, [G], [thetas], l, c_l, u_max)
    
    print('G =', G)