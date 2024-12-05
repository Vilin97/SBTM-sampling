"""Anneal the target \pi to interpolate betwenn standard gaussian and \pi"""
#%%
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

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
    t = max(t, 1e-3)
    return lambda x: 1/np.sqrt(t) * rho(x / np.sqrt(t))

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
    f = geometric_interpolation(standard_gaussian, dilation_interpolation(target_density, t), t)
    plt.plot(x, f(x), label=f't={t}')

plt.title(r'Geometric of Dilation Interpolation')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for t in t_values:
    f = dilation_interpolation(target_density, t)
    plt.plot(x, f(x), label=f't={t}')

plt.title(r'Dilation Interpolation $f(x/\sqrt{t})$')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for t in t_values:
    f = geometric_interpolation(standard_gaussian, target_density, t)
    plt.plot(x, f(x), label=f't={t}')

plt.title(r'Fisher-Rao Interpolation $f^t g^{(1-t)}$')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# for t in t_values:
#     f = lambda x : (1-t)*standard_gaussian(x) + t*target_density(x)
#     plt.plot(x, f(x), label=f't={t}')

# plt.title(r'L2 interpolation $tf + (1-t)g$')
# plt.xlabel('x')
# plt.ylabel('Density')
# plt.legend()
# plt.show()

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
prior_score = density.Density(density.gaussian_pdf, prior_params).score

target_params = {'mean': [jnp.array([-5]), jnp.array([5])], 'covariance': [jnp.array([[1.]]), jnp.array([[1.]])], 'weights': jnp.array([1/10, 1-1/10])}
target_density_obj = density.Density(density.gaussian_mixture_pdf, target_params)
target_score = target_density_obj.score
plt.plot(jnp.linspace(-10, 10, 1000), prior_density_obj.density(jnp.reshape(jnp.linspace(-10, 10, 1000), (1000,1))), label='Prior Density')
plt.plot(jnp.linspace(-10, 10, 1000), target_density_obj.density(jnp.reshape(jnp.linspace(-10, 10, 1000), (1000,1))), label='Target Density')
plt.legend()
plt.show()

# %%
def anneal_score(prior_score, target_score, t_end, annealing_schedule=lambda t: t):
    """
    interpolate between prior_score and target_score
    annealing_schedule interpolates between 0 and 1
    """
    assert annealing_schedule(1) == 1
    def annealed_score(t, x):
        t_scaled = annealing_schedule(t/t_end) # between 0 and 1
        return (1-t_scaled) * prior_score(x) + t_scaled * target_score(x)
    return annealed_score

annealing_schedule = lambda t: min((t/0.9)**2, 1)
annealed_target_score = anneal_score(prior_score, target_score, t_end, annealing_schedule)
assert jnp.allclose(annealed_target_score(0, prior_sample), prior_score(prior_sample)), "At t=0, the score should be the prior score."
assert jnp.allclose(annealed_target_score(t_end, prior_sample), target_score(prior_sample)), "At t=t_end, the score should be the target score."

# %%
# sample with sde
tol = 1e-3
# TODO: fix the syntax
# TODO: make sure the last few steps use the exact target score
annealed_scores = [lambda t, x: 1/np.sqrt(max(t, tol)) * target_score(x / np.sqrt(max(t, tol))), annealed_target_score, lambda t, x: 1/np.sqrt(max(t, tol)) * annealed_target_score(t, x / np.sqrt(max(t, tol))), target_score]

for annealed_score in annealed_scores:
    sde_logger = sampler.Logger()
    sde_sampler = sampler.SDESampler(prior_sample, annealed_score, step_size, max_steps, sde_logger)
    sde_sample = sde_sampler.sample()
    
    fig, ax = plots.plot_distributions(prior_sample, sde_sample, target_density_obj)
    ax.set_xlim(-10, 10)
    ax.set_title(fr'SDE, $\Delta t={step_size}$, $T={max_steps*step_size}$')
    fig.show()    
#%%
fig, ax = plots.plot_distributions(prior_sample, sde_sample, target_density_obj)
ax.set_xlim(-10, 10)
ax.set_title(fr'SDE, $\Delta t={step_size}$, $T={max_steps*step_size}$')
fig.show()    
# plots.visualize_trajectories(sde_logger.get_trajectory('particles'), max_time=max_steps*step_size)
# plots.plot_kl_divergence(sde_logger.get_trajectory('particles'), target_density_obj.density)

# %%
particles = sde_logger.get_trajectory('particles')
time_steps = np.linspace(0, max_steps-1, 5)
x = np.linspace(-10, 10, 1000)

plt.figure(figsize=(10, 6))
for time_step in tqdm(time_steps):
    t = time_step * step_size
    color = plt.cm.plasma(time_step / max_steps)
    f = geometric_interpolation(prior_density_obj, target_density_obj, annealing_schedule(t / t_end))
    plt.plot(x, f(x), '--', color=color, label=f'Interpolated t={t:.2f}')

    particle_array = particles[int(time_step)].reshape(-1)
    kde = gaussian_kde(particle_array, bw_method='silverman')
    plt.plot(x, kde(x), color=color, label=f'KDE t={t:.2f}')

plt.title('Interpolated Density and KDE of Particles')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

#TODO: maybe plot the error between the interpolated score and the NN, over time?