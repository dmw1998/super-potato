import numpy as np
from fenics import *

def y_l(y_L, L, gamma):
    y = np.zeros(L)
    y[-1] = y_L
    for i in range(L-1):
        l = L - 2 - i
        y[l] = (gamma ** l + gamma ** (l+1)) + y[l+1]
    return y

def find_IoQ(thetas):
    n_grid = 100
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

def sample_theta(theta_0, y, correlation=0.8):
    theta_c = correlation * theta_0 + np.sqrt(1 - correlation ** 2) * np.random.normal(0, 1, 150)
    
    g = find_IoQ(theta_c)
    
    if g < y:
        return theta_c
    else:
        return theta_0
    
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

L = 5
y = y_l(-0.01, L, 0.13)  # Adjusted final threshold
print(y)

N_initial = 10  # Increase initial sample size for better initial estimate
N_increase_factor = 2  # Factor to increase samples at each level

i = 0
N_1 = N_initial
delta = 10e6
theta_ls = []
G = []

while delta > 10e-6:
    for _ in range(N_1 - len(theta_ls)):  # Ensure we generate N_1 samples
        theta_0 = np.random.normal(0, 1, 150)
        g_0 = find_IoQ(theta_0)
        # print("IoQ:", g_0)
        theta_ls.append(theta_0)
        G.append(g_0)
    
    mask = [g < y[0] for g in G]
    p_0 = sum(mask) / N_1
    
    delta = rRMSE(p_0, mask, N_1)
    print("Level:  1", "Iteration: ", i, "p_0: ", p_0, "delta: ", delta)
    i += 1
    
p_f = p_0
    
for l in range(1, L):
    i0 = i_0(mask)
    i = 0
    N_l = N_initial
    if G[i0] < y[l]:
        p_l = 1
        delta = 0
    else:
        theta_ls = [theta_ls[i0]]
        G = [G[i0]]
        delta = 10e6
        while delta > 10e-6:
            
            if i > 2000:
                break 
            
            for _ in range(N_l - len(theta_ls)):  # Ensure we generate N_l samples
                theta_i = sample_theta(theta_ls[0], y[l])
                g_i = find_IoQ(theta_i)
                theta_ls.append(theta_i)
                G.append(g_i)
                
            mask = [g < y[l] for g in G]
            # print("mask: ", mask)
            p_l = sum(mask) / N_l
            
            delta = rRMSE(p_l, mask, N_l)
            # print("Level: ", l+1, "Iteration: ", i, "p_l: ", p_l, "delta: ", delta)
            i += 1
        
        N_l *= N_increase_factor  # Increase the number of samples for the next level if needed
        
    p_f *= p_l
    
print(p_f)
