# Approximate a lognormal random field using the Karhunen-Loeve expansion
# Since the random field is one dimensional, we can use the analytical solution

import numpy as np
import matplotlib.pyplot as plt

def kl_expan(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)
    
    # Define the KL eigenvalues and eigenfunctions for the exponential-type correlation function
    def eigenvalue(m):
        return 0.02 / np.pi ** 2 / (m + 0.5) ** 2

    def eigenfunction(m, x):
        return np.sqrt(2) * (np.sin((m + 0.5) * np.pi * x) + np.cos((m + 0.5) * np.pi * x))
    
    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m)) * eigenfunction(m, x) * thetas[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return a_x

if __name__ == "__main__":    
    # Define the correlation length
    l_c = 0.01
    
    # Define the KL coefficients
    np.random.seed(0)
    thetas = np.random.normal(0, 1, 10)
    
    np.random.seed(1)
    thetas1 = np.random.normal(0, 1, 10)
    
    # Compute the random field
    a_x = kl_expan(thetas)
    a2_x = kl_expan(thetas1)
    x = np.linspace(0, 1, 1000)
    
    # Plot the random field
    plt.plot(x,a_x)
    plt.plot(x,a2_x, 'r')
    plt.show()