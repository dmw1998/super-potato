# Input p_0, N, L_b
# Gernerate N i.i.d samples theta_1, ..., theta_N ~ f() use in MC
# Determine a failure level c_1 > 0 s.t. P_1 = p_0
# For l = 2, ..., L:
#     If l < 3, set n = 0, else set n = L_b
#     Use the N_0 failure points in F_{l-1}as seeds and generate N + (n-1)*N_0 new samples theta_i ~ f(.|F_{l-1}) with MH_type_MCMC_sampling
#     Discard the first n samples in each Markov chain
#     If l < L, determine a failure level c_l > 0 s.t. P_{l,l-1} = p_0
#     Use all N_0 failure points in F_l as seeds and generate N + (n-1)*N_0 new samples theta_k ~ f(.|F_l) with MH_type_MCMC_sampling
#     Discard the first n samples in each Markov chain
#     Evaluate P_{l-1,l} = 1/N sum_{i=1}^N I(G(theta_i) <= c_l)
# Evaluate the failure probability P_L = 1/N sum_{i=1}^N I(G(theta_i) <= c_L) and return P_F_ML = \frac{p_0^{L-1}P_{L,L-1}}{\prod_{l=2}^L P_{l-1,l}}

import numpy as np
from MH_MCMC_sampling import MH_type_MCMC_sampling

def MLE(p_0, N, L_b, gamma, M, L, f, p, G, c_1):
    # Input p_0, N, L_b
    # Gernerate N i.i.d samples theta_1, ..., theta_N ~ f() use in MC
    # Determine a failure level c_1 > 0 s.t. P_1 = p_0
    # For l = 2, ..., L:
    #     If l < 3, set n = 0, else set n = L_b
    #     Use the N_0 failure points in F_{l-1}as seeds and generate N + (n-1)*N_0 new samples theta_i ~ f(.|F_{l-1}) with MH_type_MCMC_sampling
    #     Discard the first n samples in each Markov chain
    #     If l < L, determine a failure level c_l > 0 s.t. P_{l,l-1} = p_0
    #     Use all N_0 failure points in F_l as seeds and generate N + (n-1)*N_0 new samples theta_k ~ f(.|F_l) with MH_type_MCMC_sampling
    #     Discard the first n samples in each Markov chain
    #     Evaluate P_{l-1,l} = 1/N sum_{i=1}^N I(G(theta_i) <= c_l)
    # Evaluate the failure probability P_L = 1/N sum_{i=1}^N I(G(theta_i) <= c_L) and return P_F_ML = \frac{p_0^{L-1}P_{L,L-1}}{\prod_{l=2}^L P_{l-1,l}}
    theta = np.random.normal(0, 1, N)
    p_1 = np.mean(G(theta) <= c_1)
    c = c_1
    for l in range(2, L + 1):
        if l < 3:
            n = 0
        else:
            n = L_b
        N_0 = np.sum(G(theta) <= c)
        theta = np.concatenate((theta[G(theta) <= c], MH_type_MCMC_sampling(gamma, theta[G(theta) <= c], M, N + (n-1)*N_0, f, p)))