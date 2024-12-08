#%%
"""Anneal the target \pi to interpolate betwenn standard gaussian and \pi"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

import jax
import importlib
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler, stats
from flax import nnx
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
import pickle
# reload modules
for module in [density, plots, kernel, losses, models, sampler, stats]:
    importlib.reload(module)

#%%
# visualize the interpolation between two densities

def geometric_interpolation(rho1, rho2, t):
    f = lambda x: rho1(x)**(1-t) * rho2(x)**t
    x_vals = np.linspace(-10, 10, 1000)
    f_vals = f(x_vals)
    normalization_constant = np.trapezoid(f_vals, x_vals)
    return lambda x: f(x) / normalization_constant

# from https://arxiv.org/pdf/2406.14040
def dilation_interpolation(rho, t):
    t = max(t, 0.1)
    return lambda x: 1/t * rho(x / t)

# Define the target density as a mixture of two Gaussians
def target_density(x):
    return 0.1 * jax.scipy.stats.norm.pdf(x, -5, 1) + 0.9 * jax.scipy.stats.norm.pdf(x, 5, 1)

def gaussian_interpolation(x, t):
    return 0.1 * jax.scipy.stats.norm.pdf(x, -5 * t, 1) + 0.9 * jax.scipy.stats.norm.pdf(x, 5 * t, 1)

# Define the standard Gaussian density
def standard_gaussian(x):
    return norm.pdf(x, 0, 1)

# Interpolate the densities and plot
x = np.linspace(-10, 10, 1000)
t_values = [0, 0.1, 0.25, 0.5, 0.75, 1]

plt.figure(figsize=(10, 6))
for t in t_values:
    f = gaussian_interpolation(x, t)
    plt.plot(x, f, label=f't={t}')

plt.title(r'Gaussian Interpolation')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for t in t_values:
    f = dilation_interpolation(target_density, t)
    plt.plot(x, f(x), label=f't={t}')

plt.title(r'Dilation Interpolation')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for t in t_values:
    f = geometric_interpolation(standard_gaussian, target_density, t)
    plt.plot(x, f(x), label=f't={t}')

plt.title(r'Geometric mean Interpolation $f^t g^{(1-t)}$')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

#%%
# set up
step_size = 0.01
max_steps = 100
t_end = step_size * max_steps
num_particles = 1000
key = jrandom.key(42)

prior_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[1.]])}
prior_density_obj = density.Density(density.gaussian_pdf, prior_params)
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))
# prior_score = density.Density(density.gaussian_pdf, prior_params).score
prior_density = lambda x: jax.scipy.stats.norm.pdf(x, 0, 1)
prior_score = lambda x: density.score(lambda y: prior_density(y)[0], x)


def target_density(x):
    return 0.1 * jax.scipy.stats.norm.pdf(x, -5, 1) + 0.9 * jax.scipy.stats.norm.pdf(x, 5, 1)

target_score = lambda x: density.score(lambda y: target_density(y)[0], x)


# %%
# sample with sde
# annealed_scores = [convex_combo_score, dilate_score, convex_combo_dilate_score, target_score]

#LESSON LEARNED: with higher number of steps and also bigger end time, the weights on the two mixtures get closer to the correct 0.1 and 0.9 values. E.g. 
# - with 1000 steps and 0.1 step size, get 34% and 66% weights.
# - with 100,000 steps, 0.02 steps size, dilation path, λ(t) = t^2, thresholds = (0.2, 0.9), get 23%
# - with 100,000 steps, 0.02 steps size, dilation path, λ(t) = t, thresholds = (0.2, 0.9), get 25%

def λ1(t, t_end):
    """interpolate between 0 and 1"""
    t = (t/t_end)
    return min((t/0.9), 1)

def λ2(t, t_end):
    """interpolate between 0 and 1"""
    t = (t/t_end)**2
    return min((t/0.9), 1)

def dilation_score(t, x, target_score, threshold=0.2):
    t = np.clip(t, threshold, 1)
    return target_score(x/t)

def geometric_mean_score(t, x, prior_score, target_score):
    return t * target_score(x) + (1-t) * prior_score(x)
    
#%%
# TODO: compare geometric and dilation annealing, using the metric of the number of particles < 0 (ideally, it's 10% of the total number of particles)
for (step_size, max_steps) in [(0.02, 100), (0.02, 1000), (0.02, 10000)]:#, (0.02, 100000)]:
    for (λ, lambda_name) in [(λ1, 't'), (λ2, 't^2')]:
        try:
            print(f"Lambda(t) = {lambda_name}")
            print(f"Step size={step_size}, Max steps={max_steps}, t_end={step_size * max_steps}")
            t_end = step_size * max_steps

            annealed_score = lambda t,x : dilation_score(λ(t, t_end), x, target_score)
            # annealed_score = lambda t,x : geometric_mean_score(λ(t, t_end), x, prior_score, target_score)

            sde_logger = sampler.Logger()
            sde_sampler = sampler.SDESampler(prior_sample, annealed_score, step_size, max_steps, sde_logger)
            sde_sample = sde_sampler.sample()

            # count
            particles = sde_logger.get_trajectory('particles')
            print(f"Step size={step_size}, Max steps={max_steps}")
            negative_counts = []
            for i in range(0, max_steps+1, max_steps//4):
                p = particles[i][:,0]    
                negative_count = np.sum(p < 0)
                negative_counts.append(negative_count)
                print(f"i={i}, Count of particles < 0: {negative_count}")

            #plot
            particles = sde_logger.get_trajectory('particles')
            time_steps = np.linspace(0, max_steps, 11)
            x = np.linspace(-10, 10, 1000)
            plt.figure(figsize=(10, 6))
            for idx, time_step in enumerate(tqdm(time_steps)):
                t = time_step * step_size
                color = plt.cm.plasma(time_step / max_steps)
                particle_array = particles[int(time_step)].reshape(-1)
                kde = gaussian_kde(particle_array, bw_method='silverman')
                negative_count = np.sum(particle_array < 0)
                plt.plot(x, kde(x), color=color, label=f't={t:.1f}, #<0: {negative_count}')

            plt.plot(x, target_density(x), 'r--', label='Target Density')

            plt.title('KDE of Particles at Different Time Steps')
            plt.xlabel('x')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(f'plots/annealing/dilation/kde_particles_{step_size}_{max_steps}_lambda_{lambda_name}.png')
            # plt.savefig(f'plots/annealing/geometric_mean/kde_particles_{step_size}_{max_steps}_lambda_{lambda_name}.png')
            plt.show()
        except Exception as e:
            print(f"Error: {e}")


# #%%
# particles = sde_logger.get_trajectory('particles')
# time_steps = np.linspace(0, max_steps, 11)
# x = np.linspace(-10, 10, 1000)
# plt.figure(figsize=(10, 6))
# for time_step in tqdm(time_steps):
#     t = time_step * step_size
#     color = plt.cm.plasma(time_step / max_steps)
#     particle_array = particles[int(time_step)].reshape(-1)
#     kde = gaussian_kde(particle_array, bw_method='silverman')
#     plt.plot(x, kde(x), color=color, label=f'KDE t={t:.2f}')

# plt.plot(x, target_density(x), 'r--', label='Target Density')

# plt.title('KDE of Particles at Different Time Steps')
# plt.xlabel('x')
# plt.ylabel('Density')
# plt.legend()
# plt.show()

# #%%
# particles = sde_logger.get_trajectory('particles')
# print(f"Step size={step_size}, Max steps={max_steps}")
# for i in range(0, max_steps+1, max_steps//10):
#     p = particles[i][:,0]    
#     negative_count = np.sum(p < 0)
#     print(f"i={i}, Count of particles < 0: {negative_count}")


#TODO: maybe plot the error between the interpolated score and the NN, over time?