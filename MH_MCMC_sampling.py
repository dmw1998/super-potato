# Metropolis-Hastings type MCMC sampling

# Choose a correlation parameter gamma
# Given a seed theta_1, generate a sequence of theta_i's
# repeat the following steps for n = 1, 2, ...
# Generate a candidate theta_c from a proposal distribution f(theta_c|theta_n)
# for i = 1, 2, ..., M: theta_c[i] = gamma * theta_n[i] + sqrt(1 - gamma^2) * N(0, 1)
# Accept theta_c with probability alpha(theta_n, theta_c) = min(1, p(theta_c) / p(theta_n))

# The proposal distribution f(theta_c|theta_n) is a multivariate normal distribution with mean theta_n and covariance matrix (1 - gamma^2) * I

import numpy as np

# target distribution
def target_distribution(x):
    # one dimensional Gaussian distribution
    mu = 0.0
    sigma = 1.0
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# proposal distribution
def proposal_distribution(theta_n, gamma):
    # multivariate normal distribution
    return gamma * theta_n + np.sqrt(1 - gamma**2) * np.random.normal(0, 1)

# acceptance condition
def acceptance_condition(theta_c):
    # For filter_1, the acceptance condition is theta_c <= 1 and theta_c >= 0
    return theta_c <= 1 and theta_c >= 0

# Metropolis-Hastings algorithm
def metroplis_hastings(theta, target_dist, N, gamma):
    samples = [theta]

    for _ in range(1,N):
        theta_c = proposal_distribution(theta, gamma)
        
        acceptance = acceptance_condition(theta_c)
        
        if acceptance:
            theta = theta_c

        samples.append(theta)

    return np.array(samples)