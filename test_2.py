import numpy as np
import matplotlib.pyplot as plt
from fenics import *

# Define the mean and standard deviation for the log-normal field
mu = 1.0
sigma = 0.1

# Number of terms in the truncated KL expansion
M = 150

# Define the spatial domain
x = np.linspace(0, 1, 100)

# Define the KL eigenvalues and eigenfunctions for the exponential-type correlation function
def eigenvalue(m):
    return np.sqrt(2) / np.pi / (m + 0.5)

def eigenfunction(m, x):
    return np.sin((m + 0.5) * np.pi * x)

# Generate standard Gaussian random variables theta_m
theta_m = np.random.normal(0, 1, M)

# Compute the log-normal random field log(a(x))
log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m)) * eigenfunction(m, x) * theta_m[m] for m in range(M))

# Convert to the actual random field a(x)
a_x = np.exp(log_a_x)

# Plot the random field
plt.plot(x, a_x)
plt.xlabel('x')
plt.ylabel('a(x)')
plt.title('Random Field a(x) using Truncated KL Expansion')
plt.grid(True)
plt.show()

# Define the FEniCS mesh and function space
nx = 100
mesh = IntervalMesh(nx, 0, 1)
V = FunctionSpace(mesh, 'P', 1)

# Define the random field a(x) on the FEniCS mesh
a = Function(V)
a_values = np.interp(mesh.coordinates().flatten(), x, a_x)
a.vector()[:] = a_values

# Define the boundary conditions
u_D = Constant(0.0)
def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], 0)
bc = DirichletBC(V, u_D, boundary_L)

# Define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(1.0)

a_u_x = a * u.dx(0)
a_v_x = a * v.dx(0)
F = dot(a_u_x, v.dx(0)) * dx - f * v * dx

a_form = lhs(F)
L_form = rhs(F)

# Solve the problem
u = Function(V)
solve(a_form == L_form, u, bc)

# Plot the solution
plot(u)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution to the BVP using FEniCS')
plt.grid(True)
plt.show()
