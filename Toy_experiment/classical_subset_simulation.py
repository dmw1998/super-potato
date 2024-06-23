# Problem setting:
# G ~ N(0, 1)
# G_l(\omega) = G(\omega) + \kapa(\omega) * \gamma^{l} with \kapa(\omega) ~ N(0, 1) and \gamma = 0.5
# q = 2 for the expected costs
# Prob(G <= -3.8) \approx 7.23e-05

# This script is used to simulate the classical subset selection algorithm with full-refinement

import numpy as np
import matplotlib.pyplot as plt

from Generate_G_l import *

def classical_subset_simulation(N, y_L = -3.8, p0 = 0.1, gamma = 0.5, L = 5):
    # input:
    # N: total number of samples per level
    # y_L: critical value
    # p0: probability threshold for each subset
    # gamma: auto-correlation factor
    # L: total number of levels
    
    # Initialization, l= 1
    G = np.random.normal(0, 1, N)
    kappa = np.random.choice([-1, 1], N)
    G_l = G + kappa * gamma
    
    # Compute the probability threshold
    c_l = sorted(G_l)[int(N*p0)]
    
    if c_l <= y_L:
        mask = G_l <= y_L
        return np.sum(mask) / N
    
    mask = G_l <= c_l
    G_l = G_l[mask]
    # kappa = kappa[mask]
    
    for l in range(2, L):
        G_l, kappa = Generate_G_l(G_l, kappa, N, l, c_l, gamma = gamma)
        # G_l, kappa = modified_metropolis_hastings(G_l, kappa, N, l, c_l, gamma = gamma)
        # G_l = sample_new_G(G_l, N, l, c_l)
        
        c_l = sorted(G_l)[int(N*p0)]
        # print("The probability threshold for level", l, "is", c_l)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            return p0 ** l * np.sum(mask) / N
        
        mask = G_l <= c_l
        G_l = G_l[mask]
        # kappa = kappa[mask]
        
    G_l, kappa = Generate_G_l(G_l, kappa, N, L, c_l, gamma = gamma)
    # G_l, kappa = modified_metropolis_hastings(G_l, kappa, N, l, c_l, gamma = gamma)
    # G_l = sample_new_G(G_l, N, l, c_l)
    mask = G_l <= y_L
    # print("The number of samples in the failure domain is", np.sum(mask))
    return p0 ** (L-1) * np.sum(mask) / N

if __name__ == "__main__":
    N = 70  # Total number of samples per level
    p_0 = 0.13  # Probability threshold for each subset, 0.13^5 = 7.23e-05
    gamma = 0.5
    L = 5  # Total number of levels
    y_L = -3.8  # Failure threshold
    
    np.random.seed(0)
    # p_f = classical_subset_simulation(N, p0=p_0, L=L)
    # print("The failure p_fability is {:.2e}".format(p_f))
    
    from confidence_interval import bootstrap_confidence_interval

    failure_probabilities = [classical_subset_simulation(N, p0=p_0, L=L) for _ in range(1000)]
    # print("Failure probabilities:", failure_probabilities[0:10])

    # Calculate 95% confidence interval using bootstrap method
    confidence_interval = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000, confidence_level=0.95)

    print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    # plt.xlim(1e-5, 1e-3)
    plt.step(p_f, cdf, where='post')
    plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    plt.axvline(confidence_interval[0], color='g', linestyle='--', label='95% Confidence Interval')
    plt.axvline(confidence_interval[1], color='g', linestyle='--')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.grid(True)
    plt.show()
    
    # N_ls = [50, 100, 1000, 2000, 5000, 10000]
    # p_0_ls = [0.1, 0.13, 0.15]
    # L_ls = [5, 6]
    
    # with open("subset_simulation_results.txt", "w") as f:
    #     for N in N_ls:
    #         for p_0 in p_0_ls:
    #             for L in L_ls:
    #                 f.write("N = {}, p0 = {}, L = {}\n".format(N, p_0, L))
                    
    #                 failure_probabilities = [classical_subset_simulation(N, p0=p_0, L=L, y_L=y_L) for _ in range(1000)]
                    
    #                 confidence_interval = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000, confidence_level=0.95)
                    
    #                 f.write("95% confidence interval for failure probability: ({:.2e}, {:.2e})\n\n".format(confidence_interval[0], confidence_interval[1]))