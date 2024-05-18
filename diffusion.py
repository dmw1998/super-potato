from fenics import *

def diffusion(a, n_grid):
    # -(au')' = 1 on [0,1]
    # u(0) = u'(1) = 0
    
    # a(x) is a lognormal random field
    # n_grid is the number of grid points
    
    # Create mesh and define function space
    msh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(msh, 'P', 1)

    # Define boundary condition
    u0 = Constant(0.0)
    def boundary(x):
        return x[0] < DOLFIN_EPS
    
    bc = DirichletBC(V, u0, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    
    a_form = Constant(a) * inner(grad(u), grad(v)) * dx
    L = f * v * dx

    # Compute solution
    u_h = Function(V)
    solve(a_form == L, u_h, bc)

    return u_h

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    u_max = 0.535
    N = 1000
    a = np.random.lognormal(1, 0.1, N)
    n_grid = 10
    P = np.zeros(N)
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid()
    for i in range(N):
        u = diffusion(a[i], n_grid)
        # failure: u_h(1) > 0.535
        P[i] = u(1)
        plot(u, label='a = %f' % a[i])
        
    # Plot u_max
    ax.axhline(y=u_max, color='r', linestyle='--', label='u_max')
    plt.ylim(0, u_max+0.05)
    plt.show()
    print(np.mean(P>u_max))