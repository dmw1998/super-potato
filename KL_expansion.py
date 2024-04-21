import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm

# Define the parameters of the lognormal random field
mu = 0
sigma = 0.1

# Define the covariance function
def covariance(x, y):
    return np.exp(-np.abs(x - y))

# Define the domain of the random field
x = np.linspace(0, 1, 150)

# Compute the covariance matrix
C = covariance(x[:, None], x[None, :])

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = eigh(C)

# Sort the eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Truncate the KL expansion
n_terms = 150
eigenvalues = eigenvalues[:n_terms]
eigenvectors = eigenvectors[:, :n_terms]

# Generate a sample from the lognormal random field
z = norm.rvs(size=n_terms)
y = np.exp(mu + sigma * eigenvectors @ (np.sqrt(eigenvalues) * z))

# y is now a sample from the lognormal random field