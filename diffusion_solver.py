# Solve for -(au')' = 1 on [0,1] with u(0) = 0 and u'(1) = 0
# where a is a lognormal random field approximated by KL expansion

from fenics import *
from kl_expan import kl_expan

def create_coefficient_function(a_func, V):
    # input:
    # a_func: the random field
    # V: function space
    
    # output:
    # a: coefficient function
    
    a = Function(V)
    for i, x in enumerate(V.tabulate_dof_coordinates()):
        a.vector()[i] = a_func(x[0])  # Evaluate a_func at mesh coordinates
    return a

def diffusion_solver(a_func, n_grid):
    # input:
    # a_func: the random field
    # n_grid: number of mesh points
    
    # output:
    # u_h(1): approximated IoQ
    
    # Create the mesh and define function space
    mesh = IntervalMesh(n_grid, 0, 1)
    V = FunctionSpace(mesh, 'P', 1)

    # Create coefficient function a(x)
    a = create_coefficient_function(a_func, V)

    # Define boundary condition
    u0 = Constant(0.0)
    def boundary(x):
        return x[0] < DOLFIN_EPS
    
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
    import numpy as np
    import matplotlib.pyplot as plt
    u_max = 0.535
    N = 100
    M = 150
    n_grid = 400
    G = []
    
    # Solve u_h and plot u_h
    fig, ax = plt.subplots()
    ax = plt.subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid()
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        u_h = diffusion_solver(kl_expan(thetas), n_grid)
        u_1 = u_h(1) - u_max
        G.append(u_1)
        plot(u_h, label='u_h')
        
    # Plot u_max
    ax.axhline(y=u_max, color='r', linestyle='--', label='u_max')
    plt.ylim(0, u_max+0.05)
    plt.show()