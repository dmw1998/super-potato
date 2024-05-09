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
    
    # Define the parameters of the lognormal random field
    mu = 0
    sigma = 0.1
    
    # Define the domain of the random field
    x = np.linspace(0, 1, 1000)
    
    # Define the KL expansion coefficients
    thetas = np.random.normal(0, 1, 150)
    
    # Compute the KL expansion
    a = kl_expan(thetas)
    a_values = a(x)
    
    plt.plot(x, a_values)
    plt.xlabel('x')
    plt.ylabel('a(x)')
    plt.title('Generated lognormal random field a(x)')
    plt.show()