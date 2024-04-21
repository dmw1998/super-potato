# Input p_0, N
# Gernerate N i.i.d samples theta_1, ..., theta_N ~ f() use in MC
# Determine a failure level c_1 > 0 s.t. P_1 = p_0
# where P_1 = 1/N sum_{i=1}^N I(G(theta_i) <= c_1)
# for l = 2, ..., L:
# Use the N_0 failure points in F_{l-1}as seeds and generate N - N_0 new samples theta_i ~ f(.|F_{l-1}) with MH_type_MCMC_sampling
# If l < L, determine a failure level c_l > 0 s.t. P_l = p_0
# Evaluate the failure probability P_L = 1/N sum_{i=1}^N I(G(theta_i) <= c_L) and return P_F = P_1 \times \prod_{l=2}^L P_l

import numpy as np
from MH_MCMC_sampling import metroplis_hastings

def subset_simulation(p_0, N, gamma, M, L, f, p, G, c_1):
    # Input p_0, N
    # Gernerate N i.i.d samples theta_1, ..., theta_N ~ f() use in MC
    # Determine a failure level c_1 > 0 s.t. P_1 = p_0
    # where P_1 = 1/N sum_{i=1}^N I(G(theta_i) <= c_1)
    theta = np.random.normal(0, 1, N)
    p_1 = np.mean(G(theta) <= c_1)
    c = c_1
    for l in range(2, L + 1):
        # Use the N_0 failure points in F_{l-1}as seeds and generate N - N_0 new samples theta_i ~ f(.|F_{l-1}) with metroplis_hastings
        N_0 = np.sum(G(theta) <= c)
        theta = np.concatenate((theta[G(theta) <= c], metroplis_hastings(gamma, theta[G(theta) <= c], M, N - N_0)))
        if l < L:
            # If l < L, determine a failure level c_l > 0 s.t. P_l = p_0
            c = np.quantile(theta, p)
    # Evaluate the failure probability P_L = 1/N sum_{i=1}^N I(G(theta_i) <= c_L) and return P_F = P_1 \times \prod_{l=2}^L P_l
    return p_1 * np.prod([np.mean(G(theta) <= c) for l in range(2, L + 1)])