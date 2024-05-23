# MH-type MCMC for sampling from the target distribution \phi(\cdot | F_{\ell - 1}) where \phi denotes the M-variate standard Gaussian PDF and F_{\ell} = {\theta \in \mathbb{R}^{M} : G(\theta) \leq c_{\ell}}

from fenics import *
from kl_expan import kl_expan_2
from IoQ_and_c_l import *
import numpy as np

def MCMC_sampling_each_theta(G, theta_1, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # G: the approximated IoQ
    # theta_1: current state
    # c_l: threshold value
    # u_max: critial value
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_new: new state
    
    M = len(theta_1)
    # theta_c = np.zeros_like(theta_1)
    # for i in range(M):
    #     theta_c[i] = gamma * theta_1[i] + np.sqrt(1 - gamma**2) * np.random.normal()
    theta_c = gamma * theta_1 + np.sqrt(1 - gamma**2) * np.random.randn(M)
        
    # Solve the PDE for a given mesh size
    u_h = IoQ(kl_expan_2(theta_c), n_grid)
    
    # Acceptance condition
    if u_h - u_max <= c_l:
        # If theta_c in F_{l-1}, then accept theta_c
        theta_new = theta_c
        G_new = (u_max - u_h)
    else:
        # If theta_c not in F_{l-1}, then reject theta_c
        theta_new = theta_1
        G_new = G[-1]
    
    return G_new, theta_new

def MCMC_sampling(N, G, theta_ls, u_max, c_l, n_grid, gamma = 0.8):
    # input:
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # u_max: upper bound of the solution
    # c_l: threshold value
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N - N0:
        theta = theta_ls[i]
        G_new, theta_new = MCMC_sampling_each_theta(G, theta, c_l, u_max, n_grid, gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
    
    return G, theta_ls

def MCMC_sampling_burn_in(L_b, N, G, theta_ls, c_l, u_max, n_grid, gamma = 0.8):
    # input:
    # L_b: burn-in length
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # u_max: upper bound of the solution
    # c_l: threshold value
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N + (L_b - 1) * N0:
        theta = theta_ls[i]
        G_new, theta_new = MCMC_sampling_each_theta(G, theta, c_l, u_max, n_grid, gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
        
    # Burn-in the first L_b samples for each Markov chain
    G = G[L_b * N0:]
    theta_ls = theta_ls[L_b * N0:]
    
    return G, theta_ls

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
    print('c_1 = ', c_l)
    print('G =', len(G))
    
    # Stop if the threshold value is negative
    if c_l < 0:
        return len(G) / N
    else:
        denominator = 1
        
    # l < 3, set L_b = 0
    n_grid *= 2
    c_l_1 = c_l    # c_{l-1}
    # Sampling without burn-in
    G, theta_ls = MCMC_sampling_burn_in(0, N, G, theta_ls, u_max, c_l_1, n_grid, gamma = 0.8)
    
    # Determine the threshold value c_l s.t. P_{l|l-1} = p0
    G, theta_ls, c_l = failure_level(G, theta_ls, N, 1, p0 = p0, L = L)
    print('c_2 = ', c_l)
    print('G =', len(G))
        
    if c_l < 0:
        # Stop if the threshold value is negative (reach the finest level)
        print('denominator =', denominator)
        return p0 ** l * len(G) / N / denominator
    
    # Sampling without burn-in in level l = 1
    G, theta_ls = MCMC_sampling_burn_in(0, N, G, theta_ls, u_max, c_l, n_grid, gamma = 0.8)
    
    # Eveluate P_{l-1|l}: the probability of G <= c_{l-1} (given G <= c_{l})
    G_in_l_1 = [g for g in G if g <= c_l_1]
    print('G_in_l_1 =', len(G_in_l_1))
    denominator *= len(G_in_l_1) / N
    
    # 2 < l < L, set L_b = L_b
    for l in range(2, L-1):
        n_grid *= 2
        c_l_1 = c_l     # c_{l-1}
        # Sampling with burn-in
        G, theta_ls = MCMC_sampling_burn_in(L_b, N, G, theta_ls, u_max, c_l_1, n_grid, gamma = 0.8)
        
        # Determine the threshold value c_l s.t. P_{l|l-1} = p0
        G, theta_ls, c_l = failure_level(G, theta_ls, N, l, p0 = p0, L = L)
        print('c_', l+1, '= ', c_l)
        print('G =', len(G))
        
        if c_l < 0:
            # Stop if the threshold value is negative (reach the finest level)
            print('denominator =', denominator)
            return p0 ** l * len(G) / N / denominator
        
        # Sampling with burn-in in level l
        G, theta_ls = MCMC_sampling_burn_in(L_b, N, G, theta_ls, u_max, c_l, n_grid, gamma = 0.8)
        
        # Eveluate P_{l-1|l}: the probability of G <= c_{l-1} (given G <= c_{l})
        G_in_l_1 = [g for g in G if g <= c_l_1]
        denominator *= len(G_in_l_1) / N    # * P_{l-1|l}
        
    # l = L (in program, l = L - 1)
    G, theta_ls = MCMC_sampling_burn_in(L_b, N, G, theta_ls, u_max, c_l, n_grid, gamma = 0.8)
    G, theta_ls, c_l = failure_level(G, theta_ls, N, L, p0 = p0, L = L)
    print('c_', L, ' = ', c_l)
    P_L_L_1 = len(G) / N    # P_{L|L-1}
    
    p_f = p0 ** L * P_L_L_1 / denominator
    
    return p_f

if __name__ == '__main__':
    p0 = 0.25
    M = 150
    N = 100
    L_b = 10
    print(mle(p0, M, N, L_b))