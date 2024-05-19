# Define a KL expansion for a lognormal random field

import numpy as np
import matplotlib.pyplot as plt

def kl_expan(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a function of x that represents the KL expansion of the lognormal random field
    
    M = len(thetas)
    ms = np.arange(M) + 0.5
    
    def terms(x):
        return 1 / ms * np.sin(ms * np.pi * x) * thetas
    
    def a(x):
        sums = np.sqrt(2) / np.pi * np.sum(terms(x))
        return np.exp(sums)
    
    return np.vectorize(a)

def kl_expan_2(thetas):
    # input:
    # thetas: a numpy array of length M
    
    # output:
    # a: a numpy array of length n
    
    M = len(thetas)
    
    # Define the spatial domain
    x = np.linspace(0, 1, 100)
    
    # Define the KL eigenvalues and eigenfunctions for the exponential-type correlation function
    def eigenvalue(m):
        return np.sqrt(2) / np.pi / (m + 0.5)

    def eigenfunction(m, x):
        return np.sin((m + 0.5) * np.pi * x)
    
    # Compute the log-normal random field log(a(x))
    log_a_x = 1.0 + 0.1 * sum(np.sqrt(eigenvalue(m)) * eigenfunction(m, x) * thetas[m] for m in range(M))

    # Convert to the actual random field a(x)
    a_x = np.exp(log_a_x)
    
    return a_x

import numpy as np
from scipy.optimize import fsolve

def analytical_solution(M, a, l_c):
    # input:
    # M: number of eigenvalues and eigenfunctions
    # a: parameter in the differential equation
    # l_c: parameter in the differential equation
    
    # output:
    # wn: eigenvalues
    # eigval: eigenvalues
    # eigfun: eigenfunctions
    
    c = 1 / l_c
    fun_o = lambda ww: c - ww * np.tan(ww * a)
    fun_e = lambda ww: ww + c * np.tan(ww * a)

    wn = np.zeros(M)
    eigfun = [None] * M
    
    j = 0
    k = 0

    for i in range((M + 1) // 2):
        # odd: compute data associated with equation : c - w*tan(a*w) = 0
        if (i > 0) and (2 * i - 1 < M):
            k += 1
            n = 2 * i - 1
            wn[n] = fsolve(fun_o, (k - 1) * (np.pi / a) + 1e-3)[0]
            alpha = np.sqrt(a + (np.sin(2 * wn[n] * a) / (2 * wn[n])))
            eigfun[n] = lambda x, wn=wn[n]: np.cos(wn * x) / alpha

        # even: compute data associated with equation : w + c*tan(a*w)
        if (2 * i + 2) < M:
            j += 1
            n = 2 * i + 2
            wn[n] = fsolve(fun_e, (j - 0.5) * (np.pi / a) + 1e-3)[0]
            alpha = np.sqrt(a - (np.sin(2 * wn[n] * a) / (2 * wn[n])))
            eigfun[n] = lambda x, wn=wn[n]: np.sin(wn * x) / alpha

    eigval = (2 * c) / (wn ** 2 + c ** 2)
    
    return wn, eigval, eigfun



# if __name__ == '__main__':
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import scipy.stats as stats
    
#     # Define the parameters of the lognormal random field
#     mu = 0
#     sigma = 0.1
#     M = 150
    
#     # Define the domain of the random field
#     x = np.linspace(0, 1, 100)
    
#     # Define the KL expansion coefficients
#     thetas = np.random.normal(0, 1, M)
    
#     # Compute the KL expansion
#     a = kl_expan(thetas)
#     a_values = a(x)
    
#     # Define lognormal parameters
#     mean_ln = 1
#     std_ln = 0.1

#     # Perform Kolmogorov-Smirnov Test
#     ks_stat, ks_pvalue = stats.kstest(a_values, 'lognorm', args=(std_ln,), alternative='two-sided')
#     print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_pvalue}")

#     # Histogram plot
#     plt.figure(figsize=(8, 6))
#     plt.hist(a_values, bins=30, density=True, alpha=0.6, color='g')

#     # Plot lognormal PDF for comparison
#     x = np.linspace(0, 3, 1000)
#     pdf_ln = stats.lognorm.pdf(x, std_ln, scale=np.exp(mean_ln))
#     plt.plot(x, pdf_ln, 'r-', label='Lognormal PDF')
#     plt.legend()

#     # Q-Q plot
#     plt.figure(figsize=(6, 6))
#     stats.probplot(a_values, dist="lognorm", sparams=(std_ln,), plot=plt)
#     plt.show()

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    
    # Define the parameters of the lognormal random field
    mu = 0
    sigma = 0.1
    M = 150
    
    # Define the domain of the random field
    x = np.linspace(0, 1, 100)
    
    # Define the KL expansion coefficients
    thetas = np.random.normal(0, 1, M)
    
    # Compute the KL expansion
    a = kl_expan_2(thetas)
    
    # Define lognormal parameters
    mean_ln = 1
    std_ln = 0.1

    # Perform Kolmogorov-Smirnov Test
    ks_stat, ks_pvalue = stats.kstest(a, 'lognorm', args=(std_ln,), alternative='two-sided')
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_pvalue}")

    # Histogram plot
    plt.figure(figsize=(8, 6))
    plt.hist(a, bins=30, density=True, alpha=0.6, color='g')

    # Plot lognormal PDF for comparison
    x = np.linspace(0, 3, 1000)
    pdf_ln = stats.lognorm.pdf(x, std_ln, scale=np.exp(mean_ln))
    plt.plot(x, pdf_ln, 'r-', label='Lognormal PDF')
    plt.legend()

    # Q-Q plot
    plt.figure(figsize=(6, 6))
    stats.probplot(a, dist="lognorm", sparams=(std_ln,), plot=plt)
    plt.show()