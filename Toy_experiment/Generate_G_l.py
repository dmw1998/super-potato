
# Generate G_l(\omega) = G(\omega) + \kapa(\omega) * \gamma^{l} with \kapa(\omega) ~ U({-1, 1}) and \gamma = 0.5

import numpy as np

def Generate_G_l(G_l, kappa, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    # kappa: samples in failure domain
    # l: current level
    # c_l: probability threshold for each subset
    # gamma: auto-correlation factor
    
    # output:
    # G_l: new samples for next level
    # kappa: new samples for next level
    
    N0 = len(G_l)
    # G_l = G_l - kappa * (gamma ** (l-1)) + kappa * (gamma ** l)
    
    for i in range(N - N0):
        G_l_new = 0.8 * G_l[i]  + np.sqrt(1 - 0.8 ** 2) * np.random.normal(0, 1)
        kappa_new = np.random.choice([-1, 1])
        G_l_new += kappa_new * (gamma ** l)
        
        if G_l_new <= c_l:
            # kappa = np.append(kappa, kappa_new)
            G_l = np.append(G_l, G_l_new)
        else:
            # kappa = np.append(kappa, kappa[i])
            G_l = np.append(G_l, G_l[i])
        
    return G_l, kappa

def modified_metropolis_hastings(G_l, kappa, N, l, c_l, gamma = 0.5,  proposal_std=0.8):
    # input:
    # G_l: samples in failure domain, seed
    # kappa: samples in failure domain, seed
    # N: total number of samples per level
    # l: current level
    # c_l: probability threshold for each subset
    # gamma: auto-correlation factor
    # proposal_std: standard deviation of the proposal distribution (used to control the step size of the random walk)
    
    # output:
    # G: new samples for next level
    # kappa: new samples for next level
    # G_l: new samples for next level
    
    N0 = len(G_l)
    
    for i in range(N - N0):
        # Propose a new sample for G
        G_new = np.random.normal(G_l[i], proposal_std)
        
        # Propose a new sample for kappa
        kappa_new = np.random.choice([-1, 1])
        
        # Compute the acceptance ratio
        acceptance_ratio = np.exp(-0.5 * (G_new ** 2 - G_l[i] ** 2))
        
        # Accept or reject the new sample
        if np.random.uniform(0, 1) > acceptance_ratio:
            G_new = G_l[i]
            kappa_new = kappa[i]
            
        # Compute the new G_l
        G_l_new = G_new + kappa_new * (gamma ** l)
        
        # Accept or reject the new sample
        if G_l_new <= c_l:
            kappa = np.append(kappa, kappa_new)
            G_l = np.append(G_l, G_l_new)
        else:
            kappa = np.append(kappa, kappa[i])
            G_l = np.append(G_l, G_l[i])
        
    return G_l, kappa

def sample_new_G(G_l, N, l, c_l, gamma = 0.5):
    # input:
    # G_l: samples in failure domain
    
    # output:
    # G_l: new samples for next level
    
    N0 = len(G_l)
    # for i in range(N - N0):
    #     g = np.random.normal(0, 1) + np.random.choice([-1,1]) * (gamma ** l)
        
    #     if g <= c_l:
    #         G_l = np.append(G_l, g)
    #     else:
    #         G_l = np.append(G_l, G_l[i])
    
    mask = G_l <= c_l
    
    while np.sum(mask) < N:
        G_l = G_l[mask]
        G = np.random.normal(0, 1, N-N0)
        kappa = np.random.choice([-1, 1], N-N0)
        G_l = np.append(G_l, G + kappa * (gamma ** l))
        
        mask = G_l <= c_l
            
    return G_l

if __name__ == "__main__":
    # Parameters
    N = 100  # Total number of samples per level
    p_0 = 0.1  # Probability threshold for each subset
    gamma = 0.5
    l = 1  # Level
    y_L = -3.8  # Failure threshold
    
    # Generate initial samples for level 0
    G = np.random.normal(0, 1, N)
    kappa = np.random.choice([-1, 1], N)
    G_l = G + kappa * (gamma ** l)
    
    # find c_l
    c_l = np.percentile(G_l, 100 * p_0)
    print("c_l: ", c_l)
    
    mask = G_l <= c_l
    
    G_l = G_l[mask]
    kappa = kappa[mask]
    
    l += 1
    
    # G_l, kappa = Generate_G_l(G_l, kappa, N, l, c_l, gamma = gamma)
    G_l, kappa = modified_metropolis_hastings(G_l, kappa, N, l, c_l, gamma = gamma)
    
    c_l = np.percentile(G_l, 100 * p_0)
    print("c_l: ", c_l)