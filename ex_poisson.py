# -u'' = f
# u(0) = 0, u'(1) = g

from dolfinx import *
from ufl import dx, ds, grad, inner, TrialFunction, TestFunction, Constant
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import on_boundary
from dolfinx.common import DOLFIN_EPS

# Create mesh and function space
mesh = mesh.create_unit_interval(MPI.COMM_WORLD, 4)

# Using piecewise linear continuous elements
V = fem.FunctionSpace(mesh, ("CG", 1))

# Define the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Create a function from the coefficient
u0 = 0.0
g = 1.0
f = 1.0

# Define the bilinear form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx + g * v * ds

# Define the boundary condition
def boundary(x):
    return on_boundary(mesh) and abs(x[0]) < DOLFIN_EPS

bc = fem.DirichletBC(u0)

# Solve the variational problem
u = fem.Function(V)
problem = a == L
fem.solve(problem, u, bcs=[bc])

# Plot the solution
plot(u)