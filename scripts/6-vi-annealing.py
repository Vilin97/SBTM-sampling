
"""Anneal the target \pi to interpolate betwenn standard gaussian and \pi"""

#%%
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt


def interpolated_density(rho1, rho2, t):
    f = lambda x: rho1(x)**(1-t) * rho2(x)**t
    normalization_constant, _ = quad(f, -np.inf, np.inf)
    return lambda x: f(x) / normalization_constant


# Define the target density as a mixture of two Gaussians
def target_density(x):
    return 0.1 * norm.pdf(x, -5, 1) + 0.9 * norm.pdf(x, 5, 1)

# Define the standard Gaussian density
def standard_gaussian(x):
    return norm.pdf(x, 0, 1)

# Interpolate the densities and plot
x = np.linspace(-10, 10, 1000)
t_values = [0, 0.1, 0.25, 0.5, 0.75, 1]

plt.figure(figsize=(10, 6))
for t in t_values:
    f = interpolated_density(standard_gaussian, target_density, t)
    plt.plot(x, f(x), label=f't={t}')

plt.title('Bayesian interpolation')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for t in t_values:
    f = lambda x : (1-t)*standard_gaussian(x) + t*target_density(x)
    plt.plot(x, f(x), label=f't={t}')

plt.title('L2 interpolation')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()