from fenics import *
import numpy as np
import matplotlib.pyplot as plt

class Coefficient(UserExpression):
    def __init__(self, a_vals, **kwargs):
        self.a_vals = a_vals
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = np.interp(x[0], np.linspace(0, 1, len(self.a_vals)), self.a_vals)

    def value_shape(self):
        return ()

def create_coefficient_function(a_vals, V):
    a = Function(V)
    dof_coords = V.tabulate_dof_coordinates().flatten()
    a_vals_interp = np.interp(dof_coords, np.linspace(0, 1, len(a_vals)), a_vals)
    a.vector()[:] = a_vals_interp
    return a

def diffusion(a_vals, n_grid):
    # Create mesh and define function space
    msh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(msh, 'P', 1)
    
    # Create coefficient function a(x)
    a = Coefficient(a_vals, degree=1)

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

def failure_level(G, thetas_ls, N, p0=0.1):
    sorted_indices = sorted(range(N), key=lambda k: G[k])
    sorted_G = [G[i] for i in sorted_indices]
    sorted_theta_ls = [thetas_ls[i] for i in sorted_indices]
    
    N0 = int(p0 * N)
    G = sorted_G[:N0]
    thetas_ls = sorted_theta_ls[:N0]
    c_l = sorted_G[N0-1]
    
    return G, thetas_ls, c_l

if __name__ == '__main__':
    u_max = 0.535
    N = 1000
    a_mean = 1.0
    a_sigma = 0.1
    n_grid = 10
    u_1 = np.zeros(N)
    
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid()

    for i in range(N):
        a_vals = np.random.lognormal(np.log(a_mean), a_sigma, n_grid)
        u = diffusion(a_vals, n_grid)
        u_1[i] = u(1)  # Evaluate u_h at x=1
        plot(u, label='a = %f' % a_vals[0])
        
    ax.axhline(y=u_max, color='r', linestyle='--', label='u_max')
    plt.ylim(0, u_max + 0.05)
    plt.legend()
    plt.show()
    
    print("Failure probability:", np.mean(u_1 > u_max))
