# Sampling
from sampling_MC import *

# kl expansion
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def analytical_solution(M, a, l_c):
    c = 1 / l_c
    fun_o = lambda ww: c - ww * np.tan(ww * a)
    fun_e = lambda ww: ww + c * np.tan(ww * a)

    wn = np.zeros(M)
    eigfun = [None] * M
    
    j = 0
    k = 0

    for i in range((M + 1) // 2):
        if (i > 0) and (2 * i - 1 < M):
            k += 1
            n = 2 * i - 1
            wn[n] = fsolve(fun_o, (k - 1) * (np.pi / a) + 1e-3)[0]
            alpha = np.sqrt(a + (np.sin(2 * wn[n] * a) / (2 * wn[n])))
            eigfun[n] = lambda x, wn=wn[n]: np.cos(wn * x) / alpha

        if (2 * i + 2) < M:
            j += 1
            n = 2 * i + 2
            wn[n] = fsolve(fun_e, (j - 0.5) * (np.pi / a) + 1e-3)[0]
            alpha = np.sqrt(a - (np.sin(2 * wn[n] * a) / (2 * wn[n])))
            eigfun[n] = lambda x, wn=wn[n]: np.sin(wn * x) / alpha

    eigval = (2 * c) / (wn ** 2 + c ** 2)
    
    return wn, eigval, eigfun

def generate_lognormal_random_field(mu, sigma, M, a, l_c, x):
    wn, eigval, eigfun = analytical_solution(M, a, l_c)
    theta_m = np.random.normal(0, 1, M)
    log_a_x = mu + sigma * sum(np.sqrt(eigval[m]) * eigfun[m](x) * theta_m[m] for m in range(M))
    a_x = np.exp(log_a_x)
    return a_x

# Find IoQ and c_l
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
    
    # Set PETSc options to suppress solver output
    import os
    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log -log_summary"
    
    # Compute solution
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)  # Suppress the FEniCS log messages
    solve(a_form == L, u_h, bc)
    
    return u_h(1)

# To define the failure domain
def failure_level(G, thetas_ls, N, l, p0 = 0.1, L = 6):
    # input:
    # G: approximated IoQ
    # thetas_ls: list of thetas
    # N: number of samples
    # p0: failure probability
    # L: number of levels
    
    # output:
    # G, thetas_ls: updated G and thetas_ls
    # c_l: failure level
    
    sorted_indices = sorted(range(N), key=lambda k: G[k])
    sorted_G = [G[i] for i in sorted_indices]
    sorted_theta_ls = [thetas_ls[i] for i in sorted_indices]
    
    N0 = int(p0 * N)
    c_l = sorted_G[N0-1]
    if c_l < 0 or l == L:
        # When we reach the finest level
        G = [g for g in G if g < 0]
    else:
        G = sorted_G[:N0]
        thetas_ls = sorted_theta_ls[:N0]
    
    return G, thetas_ls, c_l

# Implement subset simulation
def subset_simulation(p0, N, M, gamma = 0.8, u_max = 0.535, n_grid = 100, L = 4):
    # input:
    # p0: probability of failure
    # N: number of samples
    # M: length of theta
    # gamma: correlation parameter
    # u_max: critical value
    # n_grid: number of grid points
    # L: number of levels
    
    # output:
    # p_f: final probability of failure
    
    # Generate initial samples
    theta_ls = [np.random.randn(M) for _ in range(N)]
    G = [u_max - IoQ(kl_expan_2(theta), n_grid) for theta in theta_ls]
        
    # Determine the threshold value c_l
    G, theta_ls, c_l = failure_level(G, theta_ls, N, 0, p0 = p0, L = L)
    print('c_0 = ', c_l)
    
    if c_l < 0:
        return len(G) / N
    else:
        p_f = p0

    for l in range(1,L):
        G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma)
        G, theta_ls, c_l = failure_level(G, theta_ls, N, l, p0 = p0, L = L)
        
        print('c_l =', c_l)
        if c_l < 0:
            # When we reach the finest level
            P_L = len(G) / N
            p_f *= P_L
            break
        else:
            p_f *= p0
        
    # Compute the final probability of failure
    G, theta_ls = MCMC_sampling(N, G, theta_ls, u_max, c_l, gamma)
    G, theta_ls, c_l = failure_level(G, theta_ls, N, L, p0 = p0, L = L)
    print('c_L =', c_l)
    P_L = len(G) / N
    if P_L == 0:
        print('P_L = 0')
    else:
        print('P_L =', P_L)
        p_f *= P_L
        
    return p_f

if __name__ == '__main__':
    p0 = 0.1
    N = 1000
    M = 150
    p_f = subset_simulation(p0, N, M)
    print('Final probability of failure:', p_f)