# Metropolis-Hastings algorithm for 1D sampling

# Choose a correlation parameter gamma
# Given a seed theta_1, generate a sequence of theta_i's

import numpy as np

# # target distribution
# def target_distribution(x):
#     # one-dimensional Gaussian distribution
#     mu = 0.0
#     sigma = 1.0
#     return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# proposal distribution
def proposal_distribution(theta_n, gamma):
    # multivariate normal distribution
    return gamma * theta_n + np.sqrt(1 - gamma**2) * np.random.normal(0, 1)

# acceptance condition
def acceptance_condition(theta_c, c_min, c_max):
    # For the nested partial failure domain
    return theta_c <= c_max and theta_c >= c_min

def metropolis_hastings_1d(theta, N, gamma, c_min, c_max):
    samples = []
    for theta_1 in theta:
        samples.append(theta_1)

        for _ in range(1,N):
            theta_c = proposal_distribution(theta_1, gamma)
            
            acceptance = acceptance_condition(theta_c, c_min, c_max)
            
            if acceptance:
                theta_1 = theta_c

            samples.append(theta_1)

    return np.array(samples)

def generate_seed(x_min=0.0, x_max=1.0):
    while True:
        theta = np.random.normal(0, 1)
        if acceptance_condition(theta, x_min, x_max):
            return theta

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 10
    gamma = 0.5
    c_min = 0.0
    c_max = 1.0
    int_seed = generate_seed() # std Gaussian
    theta = [int_seed]
    samples = metropolis_hastings_1d(theta, N, gamma, c_min, c_max)
    plt.hist(samples, bins=50, density=True)
    plt.show()
