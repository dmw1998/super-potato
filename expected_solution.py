import matplotlib.pyplot as plt
import numpy as np
from fenics import *
from scipy.linalg import circulant
from matplotlib.pyplot import axhline
import scipy.sparse as sp

def generate_lognormal_field(length, num_points, l_c, mean=0, std_dev=0.1):
    """Generate a lognormal random field."""
    x = np.linspace(0, length, num_points)
    dx = x[1] - x[0]
    
    # Create covariance matrix and ensure it's symmetric positive semi-definite
    cov_matrix = np.exp(-np.abs(np.subtract.outer(x, x)) / l_c)
    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Ensure symmetry
    cov_matrix += np.eye(num_points) * 1e-10  # Ensure positive semi-definiteness
    
    gaussian_field = np.random.multivariate_normal(np.zeros(num_points), cov_matrix)
    lognormal_field = np.exp(mean + std_dev * gaussian_field)
    return x, lognormal_field

def solve_bvp_fenics(a_field, length, num_points):
    """Solve the BVP -(a(x)u'(x))' = 1 using FEniCS."""
    mesh = IntervalMesh(num_points - 1, 0, length)
    V = FunctionSpace(mesh, 'P', 1)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    a = Function(V)
    a.vector()[:] = a_field
    
    f = Constant(1.0)
    
    a_u_v = a * inner(grad(u), grad(v)) * dx
    L_v = f * v * dx
    
    bc = DirichletBC(V, Constant(0.0), 'on_boundary && near(x[0], 0.0)')
    
    u_sol = Function(V)
    solve(a_u_v == L_v, u_sol, bc)
    
    return u_sol

# Parameters
length = 1.0
num_points = 150
l_c = 0.01
num_realizations = 10000
u_max = 0.535
u_1 = []

# Prepare the plot
plt.figure(figsize=(12, 6))

for _ in range(num_realizations):
    x, lognormal_field = generate_lognormal_field(length, num_points, l_c)
    plt.subplot(2, 1, 1)
    plt.plot(x, lognormal_field, alpha=0.1)
    
    u_sol = solve_bvp_fenics(lognormal_field, length, num_points)
    u_1.append(u_sol(1) - u_max)
    
    # Extract solution at the mesh points
    u_values = u_sol.compute_vertex_values()
    x_values = np.linspace(0, length, len(u_values))
    
    plt.subplot(2, 1, 2)
    plt.plot(x_values, u_values, alpha=0.1)

plt.subplot(2, 1, 1)
plt.title("Lognormal Random Field Realizations")
plt.xlabel('x')
plt.ylabel('a(x)')

plt.subplot(2, 1, 2)
axhline(y=u_max, color='r', linestyle='--', label='u_max')
plt.title("Solutions of the Differential Equation -(a(x)u'(x))' = 1")
plt.xlabel('x')
plt.ylabel('u(x)')

plt.tight_layout()
plt.show()

# Compute the failure probability
failure_count = sum(np.array(u_1) > 0)
P = failure_count / num_realizations
print(f"Failure probability: {P}")