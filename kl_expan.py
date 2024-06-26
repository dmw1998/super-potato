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
        sums = np.sqrt(2) / np.pi * np.sum(terms(x), axis=0)
        return np.exp(sums)
    
    return np.vectorize(a)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    
    # Define the parameters of the lognormal random field
    mu = 0
    sigma = 0.1
    M = 10000
    
    # Define the domain of the random field
    x = np.linspace(0, 1, 100)
    
    # Define the KL expansion coefficients
    thetas = np.random.normal(0, 1, M)
    
    # Compute the KL expansion
    a = kl_expan(thetas)
    a_values = a(x)
    
    # Define lognormal parameters
    mean_ln = 1
    std_ln = 0.1

    # Perform Kolmogorov-Smirnov Test
    ks_stat, ks_pvalue = stats.kstest(a_values, 'lognorm', args=(std_ln,), alternative='two-sided')
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_pvalue}")

    # Histogram plot
    plt.figure(figsize=(8, 6))
    plt.hist(a_values, bins=30, density=True, alpha=0.6, color='g')

    # Plot lognormal PDF for comparison
    x = np.linspace(0, 3, 1000)
    pdf_ln = stats.lognorm.pdf(x, std_ln, scale=np.exp(mean_ln))
    plt.plot(x, pdf_ln, 'r-', label='Lognormal PDF')
    plt.legend()

    # Q-Q plot
    plt.figure(figsize=(6, 6))
    stats.probplot(a_values, dist="lognorm", sparams=(std_ln,), plot=plt)
    plt.show()