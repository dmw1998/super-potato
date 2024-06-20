import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from kl_expan import kl_expan

# Define the covariance function
def covariance(x, y, l=0.01):
    return np.exp(-np.abs(x - y) / l)

# Generate the covariance matrix
def generate_covariance_matrix(n_points, l=0.01):
    x = np.linspace(0, 1, n_points)
    cov_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            cov_matrix[i, j] = covariance(x[i], x[j], l)
    return cov_matrix, x

# Perform KL expansion
def kl_expan_covariance(n_points, M, mu=0.0, sigma=0.1, l=0.01):
    if n_points < M:
        raise ValueError(f"Number of mesh points (n_points={n_points}) must be at least equal to the number of KL terms (M={M}).")
    
    cov_matrix, x = generate_covariance_matrix(n_points, l)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = eigh(cov_matrix)
    
    # Select the largest M eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1][:M]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    def a_func(x_query, thetas):
        sums = mu + sigma * np.sum([np.sqrt(eigenvalues[i]) * eigenvectors[:, i] * thetas[i] for i in range(M)], axis=0)
        interpolated = interp1d(x, sums, fill_value="extrapolate")
        return np.exp(interpolated(x_query))
    
    return a_func


if __name__ == '__main__':
    # Compare the two methods
    M = 150
    n_points_list = [150, 200, 300]  # Ensure n_points >= M
    mu = 0.0
    sigma = 0.1

    # Generate KL expansion based on covariance matrix for each n_points
    thetas = np.random.normal(0, 1, M)

    plt.figure(figsize=(10, 8))

    for n_points in n_points_list:# First method
        a_func_1 = kl_expan(thetas)
        a_func_2 = kl_expan_covariance(n_points, M, mu, sigma)
        x = np.linspace(0, 1, 1000)  # High-resolution x for plotting
        y1 = a_func_1(x)
        y2 = a_func_2(x, thetas)
        plt.plot(x, y1, 'r', label=f'n_points = {n_points}')
        plt.plot(x, y2, 'g', label=f'n_points = {n_points}')

    plt.xlabel('x')
    plt.ylabel('a(x)')
    plt.legend()
    plt.title('KL Expansion of Lognormal Random Field with Different Mesh Points')
    plt.show()
