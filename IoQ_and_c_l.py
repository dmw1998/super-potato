# Solve for -(au')' = 1 on [0,1] with u(0) = 0 and u'(1) = 0
# where a is a lognormal random field approximated by KL expansion

from fenics import *
from kl_expan import kl_expan

def create_coefficient_function(a_func, V):
    a = Function(V)
    for i, x in enumerate(V.tabulate_dof_coordinates()):
        a.vector()[i] = a_func(x[0])  # Evaluate a_func at mesh coordinates
    return a

def IoQ(a_func, n_grid):
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

    return u_h(1)

# To define the failure domain
def failure_level(G, thetas_ls, N, p0 = 0.1):
    # input:
    # G: approximated IoQ
    # p0: failure probability
    
    # output:
    # c_l: failure level
    
    sorted_G = []
    sorted_theta_ls = []
    
    sorted_indices = sorted(range(N), key=lambda k: G[k])
    for i in sorted_indices:
        sorted_G.append(G[i])
        sorted_theta_ls.append(thetas_ls[i])
        
    N0 = int(p0 * N)
    # Keep the smallest n components and their corresponding thetas
    G = sorted_G[:N0]
    thetas_ls = sorted_theta_ls[:N0]
    c_l = sorted_G[int(np.ceil(p0 * N))]
    
    return G, thetas_ls, c_l
    

if __name__ == '__main__':
    import numpy as np
    u_max = 0.535
    N = 100
    M = 150
    n_grid = 400
    p0 = 0.1
    G = np.empty(N)
    thetas_list = np.empty((N, M))
    
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        u_1 = IoQ(kl_expan(thetas), n_grid)
        g = u_1 - u_max
        G[i] = g
        thetas_list[i] = thetas
        
        
    G, thetas, c_l = failure_level(G, thetas_list, N, p0)
    print('c_l =', c_l)
    print('G =', len(G))
    print('thetas =', len(thetas))
