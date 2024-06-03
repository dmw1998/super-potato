# Subset simulation with fully refinement

from fenics import *
from kl_expansion import *
from mh_sampling import *
from IoQ import IoQ
from failure_probability import compute_cl
import numpy as np
import matplotlib.pyplot as plt

def subset_simulation(N, M, p0, u_max, n_grid, gamma = 0.8, L = 5):
    # input:
    # N: number of required samples
    # M: number of terms in the KL expansion
    # p0: failure probability
    # u_max: upper bound of the solution
    # n_grid: number of mesh points
    # gamma: correlation parameter
    # L: number of levels
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    # Initialize the list of the approximated IoQ
    G = []
    
    # Initialize the list of the initial states
    theta_ls = []
    
    p_f = p0
    
    # Initialize the list of the initial states
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        theta_ls.append(thetas)
        u_1 = IoQ(kl_expan(thetas), n_grid)
        g = u_max - u_1
        G.append(g)
    
    # Compute the threshold value
    G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, 1, L)
    
    # print("Level: 0", "Threshold value: ", c_l)
    
    if c_l < 0:
        return len(G) / N
    
    # For l = 2, ..., L
    for l in range(1, L+1):
        # Generate N - N0 samples for each level
        G, theta_ls = sampling_theta_list(N, G, theta_ls, c_l, u_max, n_grid, gamma)
        
        # Compute the threshold value
        G, theta_ls, c_l = compute_cl(G, theta_ls, N, p0, l, L)
        
        # print("Level:", l, "Threshold value: ", c_l)
        
        if c_l < 0:
            break
        
        p_f *= p0
    
    return p_f * len(G) / N

if __name__ == "__main__":
    # Define the number of samples
    N = 100
    
    # Define the number of terms in the KL expansion
    M = 150
    
    # Define the failure probability
    p0 = 0.1
    
    # Define the upper bound of the solution
    u_max = 0.535
    
    # Define the number of mesh points
    n_grid = 100
    
    # Define the correlation parameter
    gamma = 0.8
    
    # Define the number of levels
    L = 4
    
    # np.random.seed(0)
    # # Compute the probability of failure
    # p_f = subset_simulation(N, M, p0, u_max, n_grid, gamma, L)
    # print("The probability of failure is: {:.2e}".format(p_f))
    
    np.random.seed(0)
    runs = 10
    p_f = np.zeros(runs)
    for i in range(0, runs):
        # print("Seed: ", i)
        # np.random.seed(i)
        # Compute the probability of failure
        p_f[i] = subset_simulation(N, M, p0, u_max, n_grid, gamma, L)
        print("The probability of failure is: {:.2e}".format(p_f[i]))
        
        # print("")
    
    # save p_f
    # np.save("p_f.npy", p_f)
    print("The mean of the probability of failure is: {:.2e}".format(np.mean(p_f)))