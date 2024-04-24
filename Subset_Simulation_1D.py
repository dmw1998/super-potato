from MH_MCMC_Sampling_1D import metropolis_hastings_1d
from MH_MCMC_Sampling_1D import generate_seed
from diffusion import diffusion
import numpy as np
import matplotlib.pyplot as plt

# How to determine the failure domain for the next level?
# Try 1: Cut one side of the domain - How to decide which side to cut and the cut point?
# Try 2: Cut both sides of the domain - How to decide the cut points?
# For exmaple 2, I keep the right side of the domain as the failure domain of the next level

def failure_level(theta, u, u_max, p0=0.1, x_min=0.0, x_max=1.0):
    # Cut both sides of the domain
    # keep the failure points as the boundary of failure domain of next level
    u_diff = u - u_max  # Differences u_h(1) - u_max
    c1 = np.quantile(u_diff, p0)  # Calculate the p0 quantile of u_diff, for the failure domain
    
    print(u_diff <= c1)
    print(theta)
    
    # F = {theta_i | u_h(theta_i) - u_max <= c1}
    failure_domain = theta[u_diff <= c1]
    
    x_min = np.max(theta[u_diff > c1])  # Domain contains the failure points

    return failure_domain, x_min, x_max

def subset_simulation(p0, N, L, u_max, x_min, x_max):
    # Generate N i.i.d samples theta_1, ..., theta_N ~ f() use in MC
    theta = np.zeros(N)
    for i in range(N):
        theta[i] = generate_seed(x_min, x_max)
        
    # Determine a failure level c_1 > 0 s.t. P_1 = p_0
    # Solve the diffusion equation with a = theta_i
    u = np.zeros(N)
    for i in range(N):
        u_h = diffusion(theta[i], 10)
        u[i] = u_h(1)
        
    theta, x_min, x_max = failure_level(theta, u, u_max, p0, x_min, x_max)
    print(theta)
    
    for _ in range(1, L):
        # Use the N_0 failure points in F_{l-1} as seeds and generate N - N_0 new samples theta_i ~ f(.|F_{l-1}) with MH_type_MCMC_sampling
        N_0 = len(theta)
        theta = metropolis_hastings_1d(theta, N - N_0 + 1, 0.8, x_min, x_max) # gamma = 0.8
        
        for i in range(N):
            u_h = diffusion(theta[i], 10)
            u[i] = u_h(1)
            
        theta, x_min, x_max = failure_level(theta, u, u_max, p0, x_min, x_max)
        print(theta)
        
    # Evaluate the failure probability P_L = 1/N sum_{i=1}^N I(G(theta_i) <= c_L) and return P_F = P_1 \times \prod_{l=2}^L P_l
    P_L = p0**(L-1) * np.mean(u <= u_max)
    
    return P_L
    
    

if __name__ == '__main__':
    p0 = 0.1        # Probability of failure
    N = 10          # Number of samples
    L = 10          # Number of levels
    x_min = 0.0     # Minimum value of the random field
    x_max = 1.0     # Maximum value of the random field
    u_max = 0.535   # Maximum value of the solution
    P_L = subset_simulation(p0, N, L, u_max, x_min, x_max)
    print(P_L)
