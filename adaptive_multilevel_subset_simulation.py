import numpy as np
import time
from fenics import *
set_log_level(30)

def y_l(y_L, L, gamma):
    y = np.zeros(L)
    y[-1] = y_L
    for i in range(L-1):
        l = L - 2 - i
        y[l] = (gamma ** l + gamma ** (l+1)) + y[l+1]
    return y

def find_IoQ(thetas, n_grid = 100):
    
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary conditions
    bc = DirichletBC(V, Constant(0), 'near(x[0], 0)')

    # Define the problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # mesh_points
    x = mesh.coordinates().flatten()

    # Function to define the random coefficient a(x)
    beta = 1 / 0.01
    
    def eigenvalue(m):
        w = m * np.pi
        return 2 * beta / (w**2 + beta**2)
    
    def eigenfunction(m, x):
        w = m*np.pi
        A = np.sqrt(2 * w**2 / (2*beta + w**2 + beta**2))
        B = np.sqrt(2 * beta**2 / (2*beta + w**2 + beta**2))
        return A*np.cos(w*x) + B*np.sin(w*x)
    
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1, x) * thetas[m] for m in range(len(thetas)))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)

    # Define the random field a(x) on the FEniCS mesh
    a = Function(V)
    a_values = np.interp(x, np.linspace(0, 1, len(a_x)), a_x)
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
    
    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Compute solution
    u_h = Function(V)
    # set_log_level(LogLevel.ERROR)  # Suppress the FEniCS log messages
    solve(a_form == L, u_h, bc)
    
    return 0.535 - u_h(1)

# def sample_theta(theta_0, y, correlation=0.8):
#     theta_c = correlation * theta_0 + np.sqrt(1 - correlation ** 2) * np.random.normal(0, 1, 150)
    
#     g = find_IoQ(theta_c)
    
#     if g < y:
#         return theta_c
#     else:
#         return theta_0

def sampling_one_new_theta(G, theta_1, c_l, n_grid, gamma = 0.8):
    # input:
    # G: the approximated IoQ
    # theta_1: current state
    # c_l: threshold value
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_new: new state
    
    M = len(theta_1)
    # theta_c = np.zeros_like(theta_1)
    # for i in range(M):
    #     theta = theta_1[i]
        
    #     theta_tilde = gamma * theta + np.sqrt(1 - gamma**2) * np.random.normal()
        
    #     # Compute the ratio r = f(theta_tilde) / f(theta), 
    #     # where f is the pdf of the Gaussian distribution
    #     # f(x) = exp(-0.5 * x^2) / sqrt(2 * pi)
    #     # r = exp(-0.5 * (theta_tilde**2 - theta**2))
    #     r = exp(0.5 * (theta**2 - theta_tilde**2))
        
    #     if np.random.rand() < min(1, r):
    #         theta_c[i] = theta_tilde
    #     else:
    #         theta_c[i] = theta
    
    theta_c = gamma * theta_1 + np.sqrt(1 - gamma**2) * np.random.randn(M)
        
    # Solve the PDE for required mesh size
    g_new = find_IoQ(theta_c, n_grid)
    
    # Acceptance condition
    if g_new <= c_l:
        # If theta_c in F_{l-1}, then accept theta_c
        theta_new = theta_c
        G_new = g_new
    else:
        # If theta_c not in F_{l-1}, then reject theta_c
        theta_new = theta_1
        G_new = G[-1]
    
    return G_new, theta_new

def sampling_theta_list(N, G, theta_ls, c_l, n_grid, gamma = 0.8):
    # input:
    # N: number of required samples
    # G: a list of the approximated IoQ
    # theta_ls: a list of the initial states, theta[i] has length M
    # c_l: threshold value
    # n_grid: number of mesh points
    # gamma: correlation parameter
    
    # output:
    # theta_ls: samples
    # G: approximated IoQ
    
    N0 = len(theta_ls)
    i = 0
    while i < N - N0:
        theta = theta_ls[i]
        G_new, theta_new = sampling_one_new_theta(G, theta, c_l, n_grid, gamma = gamma)
        G.append(G_new)
        theta_ls.append(theta_new)
        i += 1
    
    return G, theta_ls
    
def rRMSE(p_l, mask, N):
    if p_l == 0:
        return np.inf  # If p_l is zero, return infinity or a very large number to stop further iterations

    def R(k, p_l, mask, N):
        s = 0
        for i in range(0, N - 1 - k):
            s += mask[i] * mask[i + k]
        return s / (N - k) - p_l
    
    def rho(k, p_l, mask, N):
        r0 = R(0, p_l, mask, N)
        if r0 == 0:
            return 0
        return R(k, p_l, mask, N) / r0
    
    def gamma(p_l, mask, N):
        s = 0
        for k in range(N-1):
            s += (1 - k/N) * rho(k, p_l, mask, N)
        return 2 * s
    
    phi = gamma(p_l, mask, N)
    return np.sqrt((1 - p_l) * (1 + phi) / N)

def i_0(mask):
    for index, value in enumerate(mask):
        if value:
            return index
    return -1

# Adaptive multilevel subset simulation
def adaptive_multilevel_subset_simulation(L, y):
    M = 150
    N = 100
    n_grid = 20

    # Initialize the list of the approximated IoQ
    G = []

    # Initialize the list of the initial states
    theta_ls = []

    # Initialize the list of the initial states
    for i in range(N):
        thetas = np.random.normal(0, 1, M)
        theta_ls.append(thetas)
        g = find_IoQ(thetas, n_grid)
        G.append(g)

    mask = [g < y[0] for g in G]
    p_f = sum(mask) / N
    
    theta_new = []
    G_new = []
    for index, value in enumerate(mask):
        if value:
            theta_new.append(theta_ls[index])
            G_new.append(G[index])

    theta_ls = theta_new
    G = G_new
    n_grid *= 2
    # For l = 2, ..., L
    for l in range(1, L):
        # Generate N - N0 samples for each level
        G, theta_ls = sampling_theta_list(N, G, theta_ls, y[l-1], n_grid, 0.8)
        
        mask = [g < y[l] for g in G]
        p_l = sum(mask) / N
        
        i = 0
        while p_l == 0:
            i += 1
            if i > 200:
                print("Zero probability of failure at level: ", l, "y: ", y[l])
                return p_f
            
            # print("Zero probability of failure at level: ", l, "y: ", y[l])
            # wait for 1 second
            # time.sleep(1)
            
            theta_ls = [theta_ls[0]]
            G = [G[0]]
                
            G, theta_ls = sampling_theta_list(N, G, theta_ls, y[l-1], n_grid, 0.8)
            
            mask = [g < y[l] for g in G]
            p_l = sum(mask) / N
        
        p_f *= p_l
        n_grid *= 2
        
    return p_f

L = 6
y = y_l(0, L, 0.13)  # Adjusted final threshold
print(y)

N_initial = 10  # Increase initial sample size for better initial estimate
N_increase_factor = 2  # Factor to increase samples at each level

# i = 0
# N_1 = N_initial
# delta = 10e6
# theta_ls = []
# G = []

# while delta > 10e-6:
#     for _ in range(N_1 - len(theta_ls)):  # Ensure we generate N_1 samples
#         theta_0 = np.random.normal(0, 1, 150)
#         g_0 = find_IoQ(theta_0)
#         # print("IoQ:", g_0)
#         theta_ls.append(theta_0)
#         G.append(g_0)
    
#     mask = [g < y[0] for g in G]
#     p_0 = sum(mask) / N_1
    
#     delta = rRMSE(p_0, mask, N_1)
#     print("Level:  1", "Iteration: ", i, "p_0: ", p_0, "delta: ", delta)
#     i += 1
    
# p_f = p_0
    
# for l in range(1, L):
#     i0 = i_0(mask)
#     i = 0
#     N_l = N_initial
#     if G[i0] < y[l]:
#         p_l = 1
#         delta = 0
#     else:
#         theta_ls = [theta_ls[i0]]
#         G = [G[i0]]
#         delta = 10e6
#         while delta > 10e-6:
            
#             if i > 2000:
#                 break 
            
#             for _ in range(N_l - len(theta_ls)):  # Ensure we generate N_l samples
#                 theta_i = sample_theta(theta_ls[0], y[l])
#                 g_i = find_IoQ(theta_i)
#                 theta_ls.append(theta_i)
#                 G.append(g_i)
                
#             mask = [g < y[l] for g in G]
#             # print("mask: ", mask)
#             p_l = sum(mask) / N_l
            
#             delta = rRMSE(p_l, mask, N_l)
#             # print("Level: ", l+1, "Iteration: ", i, "p_l: ", p_l, "delta: ", delta)
#             i += 1
        
#         N_l *= N_increase_factor  # Increase the number of samples for the next level if needed
        
#     p_f *= p_l
    
# print(p_f)

np.random.seed(0)
p_f = []
for i in range(100):
    p_f.append(adaptive_multilevel_subset_simulation(L, y))
    
print(np.mean(p_f))

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    n = len(data)
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap_samples, n), replace=True)
    bootstrap_estimates = np.mean(bootstrap_samples, axis=1)
    
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    
    lower_bound = np.percentile(bootstrap_estimates, lower_percentile)
    upper_bound = np.percentile(bootstrap_estimates, upper_percentile)
    
    return lower_bound, upper_bound

# Calculate 95% confidence interval using bootstrap method
confidence_interval = bootstrap_confidence_interval(p_f, num_bootstrap_samples=100, confidence_level=0.95)

print("95% confidence interval for failure probability:", confidence_interval)