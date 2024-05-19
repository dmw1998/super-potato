# Solve for -(au')' = 1 on [0,1] with u(0) = 0 and u'(1) = 0
# where a is a lognormal random field approximated by KL expansion

from fenics import *
# from kl_expan import kl_expan
from kl_expan import kl_expan_2
import numpy as np

# class Coefficient(UserExpression):
#     def __init__(self, a_func, **kwargs):
#         self.a_func = a_func
#         super().__init__(**kwargs)

#     def eval(self, values, x):
#         values[0] = self.a_func(x[0])

#     def value_shape(self):
#         return ()

# def create_coefficient_function(a_func, V):
#     a = Function(V)
#     dof_coords = V.tabulate_dof_coordinates().flatten()
#     a_vals = np.array([a_func(x) for x in dof_coords])
#     a.vector()[:] = a_vals
#     return a

# def IoQ(a_func, n_grid):
#     # input:
#     # a_func: the random field
#     # n_grid: number of mesh points
    
#     # output:
#     # u_h(1): approximated IoQ
    
#     # Create the mesh and define function space
#     mesh = IntervalMesh(n_grid, 0, 1)
#     V = FunctionSpace(mesh, 'P', 1)

#     # Create coefficient function a(x)
#     a = Coefficient(a_func, degree=1)

#     # Define boundary condition
#     u0 = Constant(0.0)
#     def boundary(x, on_boundary):
#         return on_boundary and near(x[0], 0, DOLFIN_EPS)
    
#     bc = DirichletBC(V, u0, boundary)

#     # Define variational problem
#     u = TrialFunction(V)
#     v = TestFunction(V)
#     f = Constant(1.0)

#     a_form = inner(a * grad(u), grad(v)) * dx
#     L = f * v * dx

#     # Compute solution
#     u_h = Function(V)
#     solve(a_form == L, u_h, bc)

#     return u_h(1)

def IoQ(a_x, n_grid):
    # input:
    # a_x: the random field
    # n_grid: number of mesh points
    
    # output:
    # u_h(1): approximated IoQ
    
    # Create the mesh and define function space
    mesh = IntervalMesh(n_grid, 0, 1)
    V = FunctionSpace(mesh, 'P', 1)
    
    # Define the random field a(x) on the FEniCS mesh
    a = Function(V)
    a_values = np.interp(mesh.coordinates().flatten(), np.linspace(0, 1, len(a_x)), a_x)
    a.vector()[:] = a_values
    
    # Define boundary condition
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    
    bc = DirichletBC(V, u0, boundary)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    
    a_form = inner(a * u.dx(0), v.dx(0)) * dx
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
    # G, thetas_ls: updated G and thetas_ls
    # c_l: failure level
    
    sorted_indices = sorted(range(N), key=lambda k: G[k])
    sorted_G = [G[i] for i in sorted_indices]
    sorted_theta_ls = [thetas_ls[i] for i in sorted_indices]
    
    N0 = int(p0 * N)
    c_l = sorted_G[N0-1]
    if c_l < 0:
        # When we reach the finest level
        G = [g for g in G if g > 0]
    else:
        G = sorted_G[:N0]
        thetas_ls = sorted_theta_ls[:N0]
    
    return G, thetas_ls, c_l
    

if __name__ == '__main__':
    import numpy as np
    u_max = 0.535
    N = 1000
    M = 150
    n_grid = 400
    p0 = 0.1
    G = np.empty(N)
    thetas_list = np.empty((N, M))
    
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        u_1 = IoQ(kl_expan_2(thetas), n_grid)
        g = u_max - u_1
        G[i] = g
        thetas_list[i] = thetas
        
        
    G, thetas, c_l = failure_level(G, thetas_list, N, p0)
    print('c_l =', c_l)
    print("failure probability:", len(G) / N)
