from fenics import *
from scipy.linalg import eigh
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

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
def kl_expan_covariance(n_points, M, mu=0.0, sigma=0.1, l=0.01, thetas=None):
    if n_points < M:
        raise ValueError(f"Number of mesh points (n_points={n_points}) must be at least equal to the number of KL terms (M={M}).")

    cov_matrix, x = generate_covariance_matrix(n_points, l)

    # Eigen decomposition
    eigenvalues, eigenvectors = eigh(cov_matrix)

    # Select the largest M eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1][:M]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if thetas is None:
        thetas = np.random.normal(0, 1, M)

    def a_func(x_query):
        sums = mu + sigma * np.sum([np.sqrt(eigenvalues[i]) * eigenvectors[:, i] * thetas[i] for i in range(M)], axis=0)
        interpolated = interp1d(x, sums, fill_value="extrapolate")
        return np.exp(interpolated(x_query))

    return a_func

def create_coefficient_function(a_vals, V):
    # input:
    # a_vals: coefficient values
    # V: function space
    
    # output:
    # a: coefficient function
    
    a = Function(V)
    dof_coords = V.tabulate_dof_coordinates().flatten()
    a_vals_interp = np.interp(dof_coords, np.linspace(0, 1, len(a_vals)), a_vals)
    a.vector()[:] = a_vals_interp
    return a

def diffusion_solver(a_vals, n_grid):
    # input:
    # a_vals: coefficient values
    # n_grid: number of mesh points
    
    # output:
    # u_h(1): approximated IoQ
    
    # Create the mesh and define function space
    mesh = IntervalMesh(n_grid, 0, 1)
    V = FunctionSpace(mesh, 'P', 1)

    # Create coefficient function a(x)
    a = create_coefficient_function(a_vals, V)

    # Define boundary condition
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)

    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx

    # Compute solution
    u_h = Function(V)
    solve(a_form == L, u_h, bc)

    return u_h

if __name__ == '__main__':
    u_max = 0.535
    N = 1
    M = 150
    n_grid = 400
    G = []
    
    # Solve u_h and plot u_h
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid()
    
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        kl_expan = kl_expan_covariance(n_grid, M, mu=0, sigma=1, thetas=thetas)
        a_vals = kl_expan(np.linspace(0, 1, n_grid))
        
        # Plotting a(x) for debugging
        plt.figure()
        plt.plot(np.linspace(0, 1, n_grid), a_vals, label=f'a(x)_{i+1}')
        plt.legend()
        plt.show()
        
        u_h = diffusion_solver(a_vals, n_grid)
        u_1 = u_h(1) - u_max
        G.append(u_1)
        plot(u_h, label=f'u_h_{i+1}')

    # Plot u_max
    ax.axhline(y=u_max, color='r', linestyle='--', label='u_max')
    plt.ylim(0, u_max+0.05)
    plt.legend()
    plt.show()
