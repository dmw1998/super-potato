# MH-type MCMC for sampling from the target distribution \phi(\cdot | F_{\ell - 1}) where \phi denotes the M-variate standard Gaussian PDF and F_{\ell} = {\theta \in \mathbb{R}^{M} : G(\theta) \leq c_{\ell}}

from fenics import *
from kl_expan import kl_expan
import numpy as np

def MCMC_sampling_each_theta(G, theta_1, c_l, u_max, gamma = 0.8):
    # input:
    # G: the approximated IoQ
    # theta_1: current state
    # c_l: threshold value
    # u_max: critial value
    # gamma: correlation parameter
    
    # output:
    # theta_new: new state
    
    M = len(theta_1)
    theta_c = np.zeros_like(theta_1)
    for i in range(M):
        theta_c[i] = gamma * theta_1[i] + np.sqrt(1 - gamma**2) * np.random.normal()
        
    u_h = IoQ(kl_expan(theta_c), 4)
    
    if u_h - u_max <= c_l:
        theta_new = theta_c
        G_new = (u_h - u_max)
    else:
        theta_new = theta_1
        G_new = G[-1]
    
    return G_new, theta_new

def MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma = 0.8):
    # input:
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # u_max: upper bound of the solution
    # c_l: threshold value
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N - N0:
        theta = theta_ls[i]
        G_new, theta_new = MCMC_sampling_each_theta(G, theta, c_l, u_max, gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
    
    return G, theta_ls


import unittest
import numpy as np
from IoQ_and_c_l import *

class TestMCMCSampling(unittest.TestCase):
    def test_MCMC_sampling(self):
        # Define the parameters for the MCMC_sampling function
        N = 10
        M = 1
        c_l = 5
        u_max = 0.535
        gamma = 0.8
        G = []
        theta_ls = []
        
        # Initial a theta list and corresponding G
        for i in range(3):
            thetas = np.random.normal(0, 1, M)
            theta_ls.append(thetas)
            
        print(len(theta_ls))
        
        for thetas in theta_ls:
            g = IoQ(kl_expan(thetas), 4) - u_max
            G.append(g)

        # Call the MCMC_sampling function
        G, thetas_list = MCMC_sampling(N, G, theta_ls, c_l, u_max, gamma)

        # Check that the results is a list
        self.assertIsInstance(G, list)
        self.assertIsInstance(thetas_list, list)

        # Check that the length of the results is N
        self.assertEqual(len(G), N)
        self.assertEqual(len(thetas_list), N)

        # Check that the elements of the result are floats
        for element in G:
            self.assertIsInstance(element, float)
            
        # Check that the elements of the thetas_list are array of length M
        for element in thetas_list:
            self.assertIsInstance(element, np.ndarray)
            self.assertEqual(len(element), M)

if __name__ == '__main__':
    unittest.main()