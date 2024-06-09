# An idea

from fenics import *
from compute_yl import y_l
from kl_expansion import *
from mh_sampling import *
from IoQ import IoQ
import numpy as np

def subset_simulation_yl(N, M, u_max, n_grid, gamma, corr_coeff = 0.8, L = 5):
    # input:
    # N: number of required samples
    # M: number of terms in the KL expansion
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: accuracy parameter
    # corr_coeff: correlation parameter
    # L: number of levels
    
    # output:
    # p_f: failure probability
    
    s = True
    
    # Initialize the threshold values
    y = y_l(L, gamma)
    
    # Initialize the list of the approximated IoQ
    G = []
    
    # Initialize the list of the initial states
    theta_ls = []
    
    # Initialize the list of the initial states
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        u_1 = IoQ(kl_expan(thetas), n_grid)
        g = u_max - u_1
        if g < y[0]:
            theta_ls.append(thetas)
            G.append(g)
            
    p_f = len(G) / N
    # print("p_f: ", p_f)
            
    for l in range(1, L):
        # Generate N - N0 samples for each level
        G_c, theta_c_ls = sampling_theta_list(N, G, theta_ls, y[l-1], u_max, n_grid, corr_coeff)
        
        theta_ls = []
        G = []
        for i in range(N):
            g = G_c[i]
            if g < y[l]:
                theta_ls.append(theta_c_ls[i])
                G.append(g)
                
        if len(G) == 0:
            print("No sample in the failure domain F_{", l+1, "}")
            s = False
            break
        
        # print(len(G) / N)
        p_f *= len(G) / N
        
    return s, p_f

if __name__ == "__main__":
    N = 100
    M = 150
    u_max = 0.535
    n_grid = 100
    gamma = 0.15
    corr_coeff = 0.8
    L = 5
    
    # np.random.seed(1)
    # s, p_f = subset_simulation_yl(N, M, u_max, n_grid, gamma, corr_coeff, L)
    # print("failure probability{:.2e}".format(p_f))
    s = False
    p_f = np.zeros(100)
    for i in range(100):
        ind = i
        while True:
            s, p = subset_simulation_yl(N, M, u_max, n_grid, gamma, corr_coeff, L)
            if s:
                p_f[ind] = p
                print("failure probability {:.2e}".format(p_f[ind]))
                break
        
    p_f = np.mean(p_f)
    print("failure probability {:.2e}".format(p_f))