# Find the failure probability with multilevel Monte Carlo with selective refinement

from fenics import *
from selective_refinement import *

def failure_level(G, thetas_ls, N, l, p0=0.1, L=6):
    # Sort G and corresponding thetas
    sorted_indices = sorted(range(len(G)), key=lambda k: G[k])
    sorted_G = [G[i] for i in sorted_indices]
    sorted_theta_ls = [thetas_ls[i] for i in sorted_indices]

    N0 = int(p0 * N)
    c_l = sorted_G[N0-1]

    G = []
    thetas_ls = []
    if c_l < 0 or l == L:
        for i, g in enumerate(sorted_G):
            if g < 0:
                G.append(sorted_G[i])
                thetas_ls.append(sorted_theta_ls[i])
    else:
        G = sorted_G[:N0]
        thetas_ls = sorted_theta_ls[:N0]

    return G, thetas_ls, c_l


def sus_sr(p0, N, M, gamma = 0.8, u_max = 0.535, L = 6):
    # input:
    # p0: failure probability
    # N: number of samples
    # M: length of theta
    # gamma: correlation parameter
    # u_max: critical value
    # L: largest number of levels (approximated by failure probability)
    
    # output:
    # p_f: final probability of failure
    
    # Initialization
    c_0 = 1    # a large threshold value
    theta_ls = [np.random.randn(M) for _ in range(N)]
    G = []
    for theta in theta_ls:
        G_i = selective_refinement(0, theta, u_max, c_0)
        G.append(G_i)
        
    # Determine the threshold value c_l
    # print(G)
    G, theta_ls, c_l = failure_level(G, theta_ls, N, 0, p0 = p0, L = L)
    print('c_0 =', c_l)
    
    if c_l < 0:
        return len(G) / N
    else:
        p_f = p0
        
    for l in range(1, L):
        G, theta_ls = MCMC_sampling_sr(N, G, theta_ls, u_max, c_l, gamma)
        c_l_1 = c_l
        print(G)
        G, theta_ls, c_l = failure_level(G, theta_ls, N, l, p0 = p0, L = L)
        print('c_', l, ' =', c_l)
        
        if c_l < 0:
            P_L = len(G) / N
            p_f *= P_L
            break
        else:
            p_f *= p0
            
    # Compute the final probability of failure
    G, theta_ls = MCMC_sampling_sr(N, G, theta_ls, u_max, c_l, gamma)
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
    N = 10
    M = 150
    p_f = sus_sr(p0, N, M)
    print('Final probability of failure =', p_f)