# Solve the toy experiment using the subset simulation method with the selective refinement.

import numpy as np
import time
import matplotlib.pyplot as plt

from genrate_y_l import y_l
from Generate_G_l import *

def subset_simulation_sr(L, gamma, y_L, N):
    # input:
    # L: number of levels
    # gamma: accuracy parameter s.t. |G - G_l| <= gamma^{l}
    # y_L: the value of y_L
    # N: the number of samples per level
    
    # output:
    # p_f: the probability of failure
    # p_f_hat: the probability of failure estimated by the subset simulation
    # p_f_hat_sr: the probability of failure estimated by the subset simulation with the selective refinement
    
    # To compute the sequence of failure thresbolds y_l
    # from genrate_y_l import y_l
    y = [-1.3, -2, -2.8, -3.3, y_L]
    
    # To generate the samples
    while True:
        G = np.random.normal(0, 1, N)
        kappa = np.array([k(g) for g in G])
        G_l = G + kappa * gamma
        
        mask = G_l <= y[0]
        # print(mask.sum())
        
        if mask.sum() > 0:
            p_f = mask.mean()
            G_l = G_l[mask][:1]
            break
        
    # for l in range(1, L+1):
    #     i = 0
    #     while i < 5000:
    #         i += 1
    #         G_l = sample_new_G(G_l, N, l, y[l-1], gamma)
            
    #         mask = G_l <= y[l]
    #         # print(mask.sum())
            
    #         if mask.sum() > 0:
    #             p_f *= mask.mean()
    #             G_l = G_l[mask][:1]
    #             break
            
    #     print("Level: ", l, "Number of iterations: ", i)
    
    for l in range(1, L):
        while True:
            # print("Level: ", l+1)
            # time.sleep(0.5)
            G_l = sample_new_G(G_l, N, l+1, y[l-1], gamma)
            
            mask = G_l <= y[l]
            G_l = G_l[mask]
            
            if mask.sum() > 0:
                p_f *= mask.mean()
                break
                
    return p_f

if __name__ == "__main__":
    L = 5
    gamma = 0.5
    y_L = -3.8
    N = 1000
    
    np.random.seed(0)
    
    failure_probabilities = [subset_simulation_sr(L, gamma, y_L, N) for _ in range(1000)]
    print("The mean of the failure probability: ", np.mean(failure_probabilities))
    
    from confidence_interval import bootstrap_confidence_interval
    
    # Calculate 95% confidence interval using bootstrap method
    confidence_interval, ci = bootstrap_confidence_interval(failure_probabilities, num_bootstrap_samples=1000)

    print("95% confidence interval for failure probability: ({:.2e}, {:.2e})".format(confidence_interval[0], confidence_interval[1]))
    
    print("90% confidence interval for failure probability: ({:.2e}, {:.2e})".format(ci[0], ci[1]))
    
    p_f = sorted(failure_probabilities)
    cdf = np.arange(1, len(p_f) + 1) / len(p_f) 

    # Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.step(p_f, cdf, where='post')
    plt.axvline(7.23e-05, color='r', linestyle='--', label='True Value')
    plt.axvline(confidence_interval[0], color='g', alpha = 0.5, linestyle='--', label='95% Confidence Interval')
    plt.axvline(confidence_interval[1], color='g', alpha = 0.5, linestyle='--')
    plt.axvline(ci[0], color='m', alpha = 0.5, linestyle='--', label='90% Confidence Interval')
    plt.axvline(ci[1], color='m', alpha = 0.5, linestyle='--')
    plt.xlabel('Probability')
    plt.ylabel('Empirical CDF')
    plt.title('Empirical CDF of Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()
