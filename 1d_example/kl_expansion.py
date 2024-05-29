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
    
    # def eigenvalue(m):
    #     return 0.01 / np.pi / (m + 0.5)

    # def eigenfunction(m, x):
    #     if m % 2 == 0:
    #         return np.sqrt(2) * np.cos((m + 0.5) * np.pi * x)
    #     else:
    #         return np.sqrt(2) * np.sin((m + 0.5) * np.pi * x)
        
    # def eigenfunction(m, x):
    #     return np.sqrt(2) * np.sin((m + 0.5) * np.pi * x)
    
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


def kl_expansion_corr(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)
    
    # Define the covariance function based on the correlation length
    def covariance(x1, x2):
        return np.exp(-np.abs(x1 - x2) / 0.01)
    
    # Create the covariance matrix
    cov_matrix = np.array([[covariance(xi, xj) for xj in x] for xi in x])

    # Eigenvalue decomposition to find eigenvalues and eigenfunctions
    eigenvalues, eigenfunctions = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenfunctions in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx][:M]
    eigenfunctions = eigenfunctions[:, idx][:, :M]
    
    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))

    # Construct log(a(x)) using the KL expansion
    log_a_x = mu + sigma * np.sum([np.sqrt(eigenvalues[n]) * eigenfunctions[:, n] * thetas[n] for n in range(M)], axis=0)

    # Construct a(x)
    a_x = np.exp(log_a_x)
    
    return a_x

if __name__ == "__main__":    
    # Define the correlation length
    l_c = 0.01
    
    M = 1
    
    # Define the KL coefficients
    np.random.seed(0)
    thetas = np.random.normal(0, 1, M)
    
    # Compute the random field
    a_x = kl_expan(thetas)
    a_x_corr = kl_expansion_corr(thetas)
    x = np.linspace(0, 1, 1000)
    
    # Plot the random field
    plt.plot(x,a_x)
    plt.plot(x,a_x_corr)
    plt.show()