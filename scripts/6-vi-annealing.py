"""Anneal the target \pi to interpolate betwenn standard gaussian and \pi"""
#%%
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
    t = max(t, 1e-3)
    return lambda x: 1/np.sqrt(t) * rho(x / np.sqrt(t))

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
    f = geometric_interpolation(standard_gaussian, dilation_interpolation(target_density, t), t)
    plt.plot(x, f(x), label=f't={t}')

plt.title(r'Geometric mean of Dilation Interpolation')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 6))
# for t in t_values:
#     f = dilation_interpolation(target_density, t)
#     plt.plot(x, f(x), label=f't={t}')

# plt.title(r'Dilation Interpolation $f(x/\sqrt{t})$')
# plt.xlabel('x')
# plt.ylabel('Density')
# plt.legend()
# plt.show()

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
step_size = 0.1
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
def dilate(f, tol=5e-2):
    return lambda t, x: 1/np.sqrt(max(t, tol)) * f(x / np.sqrt(max(t, tol)))

def convex_combo(a, b):
    return lambda t: (1-t)*a + t*b

def λ(t):
    """interpolate between 0 and 1"""
    t = t/t_end
    return min((t/0.5), 1)

convex_combo_score = lambda t, x: convex_combo(prior_score(x), target_score(x))(λ(t))
dilate_score = lambda t, x: dilate(target_score)(λ(t), x)
convex_combo_dilate_score = lambda t, x: convex_combo(prior_score(x), dilate_score(t, x))(λ(t))
#%%
time_points = np.linspace(0, t_end, 5)[1:-1]  # 3 time points between 0 and t_end, without end-points
x = jnp.linspace(-2.5, 2.5, 1000)

plt.figure(figsize=(10, 6))
line_styles = ['-', '--', '-.']
colors = ['b', 'g', 'r']
for idx, t in enumerate(time_points):
    line_style = line_styles[idx % len(line_styles)]
    
    convex_combo_score_vals = convex_combo_score(t, x)
    dilate_score_vals = dilate_score(t, x)
    convex_combo_dilate_score_vals = convex_combo_dilate_score(t, x)
    target_score_vals = target_score(x)

    plt.plot(x, convex_combo_score_vals, line_style, color=colors[0], label=f'Convex Combo Score at t={t:.2f}')
    plt.plot(x, dilate_score_vals, line_style, color=colors[1], label=f'Dilate Score at t={t:.2f}')
    plt.plot(x, convex_combo_dilate_score_vals, line_style, color=colors[2], label=f'Convex Combo Dilate Score at t={t:.2f}')

plt.plot(x, target_score_vals, '-', color='black', linewidth=2, label='Target Score')
plt.title('Scores at Different Time Points')
plt.xlabel('x')
plt.ylabel('Score')
plt.legend()
plt.show()
# %%
# sample with sde
# annealed_scores = [convex_combo_score, dilate_score, convex_combo_dilate_score, target_score]
annealed_scores = [lambda t, x: density.score(lambda y: gaussian_interpolation(y, λ(t))[0], x)]

for annealed_score in annealed_scores:
    sde_logger = sampler.Logger()
    sde_sampler = sampler.SDESampler(prior_sample, annealed_score, step_size, max_steps, sde_logger)
    sde_sample = sde_sampler.sample()
    
    fig, ax = plots.plot_distributions(prior_sample, sde_sample, target_density_obj)
    ax.set_xlim(-10, 10)
    ax.set_title(fr'SDE, $\Delta t={step_size}$, $T={max_steps*step_size}$')
    fig.show()    
    
    
#%%
# Define different lambda functions
lambdas = [
    lambda t: min((t / 0.5), 1),
    lambda t: t,
    lambda t: t**2,
    lambda t: np.sqrt(t)
]

# Define annealing variants
annealing_variants = {
    'convex_combo': convex_combo_score,
    'dilate': dilate_score,
    'convex_combo_dilate': convex_combo_dilate_score,
    'none': target_score
}

results = {}

# Run experiments
for lambda_func in lambdas:
    for name, annealed_score in annealing_variants.items():
        sde_logger = sampler.Logger()
        sde_sampler = sampler.SDESampler(prior_sample, annealed_score, step_size, max_steps, sde_logger)
        sde_sample = sde_sampler.sample()
        
        # Save particles
        particles = sde_logger.get_trajectory('particles')
        filename = f'particles_{name}_lambda_{lambda_func.__name__}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(particles, f)
        
        # Compute KL divergence
        kl_div = stats.kl_divergence(sde_sample, target_density_obj.density)
        results[(name, lambda_func.__name__)] = kl_div

# Find the best setting
best_setting = min(results, key=results.get)
print(f'Best setting: {best_setting} with KL divergence: {results[best_setting]}')

# Load and plot the best particles
best_name, best_lambda = best_setting
filename = f'particles_{best_name}_lambda_{best_lambda}.pkl'
with open(filename, 'rb') as f:
    best_particles = pickle.load(f)

time_steps = np.linspace(0, max_steps-1, 5)
x = np.linspace(-10, 10, 1000)

plt.figure(figsize=(10, 6))
for time_step in tqdm(time_steps):
    t = time_step * step_size
    color = plt.cm.plasma(time_step / max_steps)
    f = geometric_interpolation(prior_density_obj, target_density_obj, annealing_schedule(t / t_end))
    plt.plot(x, f(x), '--', color=color, label=f'Interpolated t={t:.2f}')

    particle_array = best_particles[int(time_step)].reshape(-1)
    kde = gaussian_kde(particle_array, bw_method='silverman')
    plt.plot(x, kde(x), color=color, label=f'KDE t={t:.2f}')

plt.title('Interpolated Density and KDE of Particles')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

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