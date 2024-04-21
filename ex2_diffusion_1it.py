# -(au')' = 1 for u = u(x), x \in [0,1]
# BC: u(0) = u'(1) = 0
# a is a lognormal random field with constant mean value mu_a = 1 and std deviation sigma_a = 0.1
# the covariance function is given by: C(x,y) = exp(-|x-y|/lambda) with lambda = 0.01
# The spatial discretization is done using a finite element method with piecewise linear continuous elements
# The system response is Q = u(1), Q_h = u_h(1) where u_h is the numerical solution, u_h(1) > u_max = 0.535
# Associated failure probability is P_Fh = 1.6e-4

import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm
import matplotlib.pyplot as plt
import ufl
from dolfinx import fem, io, mesh, geometry
import dolfinx.fem.petsc
from ufl import dx, grad, inner
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl.coefficient
import ufl.constant

# problem:
# $$ -(au')' = 1, \quad u(0) = u'(1) = 0 $$

# a is a lognormal random field with constant mean value mu_a = 1 and std deviation sigma_a = 0.1
# the covariance function is given by: C(x,y) = exp(-|x-y|/lambda) with lambda = 0.01
# The spatial discretization is done using a finite element method with piecewise linear continuous elements
# The system response is Q = u(1), Q_h = u_h(1) where u_h is the numerical solution, u_h(1) > u_max = 0.535
# Associated failure probability is P_Fh = 1.6e-4

# Set random seed for reproducibility
np.random.seed(0)

M = 150

# Give a correlation parameter gamma
gamma = 0.8

# Given a seed theta_1, generate a sequence of theta_i's
theta_1 =np.random.normal(1, 0.1)
theta = [theta_1]

for i in range(1, M):
    theta_c = gamma * theta_1 + np.sqrt(1 - gamma**2) * np.random.normal(0, 1)
    
    if theta_c <= 1 and theta_c >= 0:
        theta_1 = theta_c
        
    theta.append(theta_1)
    
# Define the parameters of the lognormal random field
mu = 0
sigma = 0.1
lambda_a = 0.01

# Define the covariance function
def covariance(x, y):
    return np.exp(-np.abs(x - y)/lambda_a)

# Define the domain of the random field
x = np.linspace(0, 1, M)

# Compute the covariance matrix
C = covariance(x[:, None], x[None, :])

# Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = eigh(C)

# Sort the eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Truncate the KL expansion
eigenvalues = eigenvalues[:M]
eigenvectors = eigenvectors[:, :M]

# Generate a sample from the lognormal random field
a = np.exp(mu + sigma * eigenvectors @ (np.sqrt(eigenvalues) * theta))

# To determine the failure level c_1 > 0 s.t. P_1 = p_0
# on each level
p_0 = 0.1

# 500 runs of the simulation
tol_runs = 500

# total number of samples on each level
N = 1000

# Define the mesh and function space
# Start from h_1 = 1/4
comm = MPI.COMM_WORLD
mesh = mesh.create_unit_interval(comm, 4)

# Using piecewise linear continuous elements
V = fem.FunctionSpace(mesh, ("P", 1))

# Define the trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Create a function from the coefficient
a_form = -a * inner(grad(u), grad(v)) * dx
L = 1 * v * dx

# Define the boundary conditions
facets = mesh.locate_entities_boundary(
    mesh,
    dim = mesh.topology.dim - 1,
    marker = lambda x: np.isclose(x[0], 0.0),
)

dofs = fem.locate_dofs_geometrical(V=V, entity_dim = 1, entities = facets)

bcs = fem.dirichletbc(value = ScalarType(0), dofs = dofs, V=V)

# solve the problem
u = fem.Function(V)
problem = ufl.Form(a_form)
problem += ufl.Form(L)
fem.solve(problem == 0, u, bcs=bcs)

# Compute the system response
Q = u(1)

# Define the failure level
u_max = 0.535

# Compute the failure probability
P_F = np.mean(Q <= u_max)

print(f"Failure probability: {P_F}")