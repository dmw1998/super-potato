import numpy as np
import matplotlib.pyplot as plt
import scipy

# Define the kl_expan function
def kl_expan(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 1000)
    
    # Define the KL eigenvalues and eigenfunctions for the exponential-type correlation function
    def eigenvalue(m):
        return 1 / np.pi ** 2 / (m + 0.5) ** 2

    def eigenfunction(m, x):
        return np.sqrt(2) * (np.sin((m + 0.5) * np.pi * x) + np.cos((m + 0.5) * np.pi * x))
    
    # Compute the mean and standard deviation
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    # Compute the log-normal random field log(a(x))
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m)) * eigenfunction(m, x) * thetas[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return x, a_x

# Define the function to compute u(1)
def compute_u1(thetas):
    x, a_x = kl_expan(thetas)
    a_1 = a_x[-1]  # a(1) is the last value in the array a_x

    # Define the integrand function
    integrand = (x - 1 - a_1) / a_x

    # Compute the integral using the trapezoidal rule or Simpson's rule
    u1 = -scipy.integrate.simpson(y=integrand, x=x)

    return u1

# Example usage:
np.random.seed(0)
thetas = np.random.randn(150)  # Example thetas
x, a_x = kl_expan(thetas)
plt.plot(x, a_x)
plt.show()
u1 = compute_u1(thetas)
print(f"u(1) = {u1}")

# To compute the probability that u(1) > 0.535
N_samples = 1000
count_above_threshold = 0
u1_values = []

for _ in range(N_samples):
    thetas = np.random.randn(150)  # Generate new sample of thetas
    u1 = compute_u1(thetas)
    u1_values.append(u1)
    if u1 > 0.535:
        count_above_threshold += 1

# Estimate the probability
probability = count_above_threshold / N_samples
print(f"The estimated probability that u_1 > 0.535 is: {probability}")

# Optionally plot the histogram of u1 values

plt.hist(u1_values, bins=50, alpha=0.75)
plt.axvline(x=0.535, color='r', linestyle='--')
plt.xlabel('u(1)')
plt.ylabel('Frequency')
plt.title('Histogram of u(1) values')
plt.show()

# import numpy as np
# from scipy.optimize import fsolve
# import matplotlib.pyplot as plt

# def kl_expan2(thetas, L=0.1, n_points=1000):
#     M = len(thetas)
#     x = np.linspace(0, 1, n_points)  # Discretize the interval [0, 1]
    
#     # Solve for omega_n using the transcendental equation
#     def transcendental_eqn(omega, L):
#         return np.tan(omega) - (2 * L * omega) / (L**2 * omega**2 + 1)
    
#     omegas = np.zeros(M)
#     initial_guesses = (np.arange(M) + 0.5) * np.pi  # Initial guesses for omega_n
    
#     for i in range(M):
#         omegas[i] = fsolve(transcendental_eqn, initial_guesses[i], args=(L))[0]

#     # Compute the eigenvalues
#     thetas_n = np.array([2 * L / (L**2 * omega**2 + 1) for omega in omegas])
    
#     # Compute the eigenfunctions
#     def eigenfunction(n, x, L, omegas):
#         omega = omegas[n]
#         A = np.sqrt(2 / (1 + L**2 * omega**2))
#         return A * (np.sin(omega * x) + L * omega * np.cos(omega * x))

#     # Mean and standard deviation for log-normal distribution
#     mu = -0.5 * np.log(1.01)
#     sigma = np.sqrt(np.log(1.01))
    
#     # Generate the log-normal random field log(a(x))
#     log_a_x = mu + sigma * sum(np.sqrt(thetas_n[n]) * eigenfunction(n, x, L, omegas) * thetas[n] for n in range(M))

#     # Convert to the actual random field a(x)
#     a_x = np.exp(log_a_x)
    
#     return x, a_x

# def test_kl_expan():
#     from kl_expansion import kl_expan
#     np.random.seed(0)
#     thetas = np.random.randn(1)  # Generate random coefficients
#     x, a_x = kl_expan2(thetas, L=0.01)  # Compute the random field
    
#     # Compute the random field
#     a2_x = kl_expan(thetas)
    
#     # Plot the result
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, a_x, label='a_1(x)')
#     plt.plot(x, a2_x, label='a_2(x)')
#     plt.xlabel('x')
#     plt.ylabel('a(x)')
#     plt.title('KL Expansion of the Random Field a(x)')
#     plt.legend()
#     plt.show()

# # Run the test function
# test_kl_expan()
