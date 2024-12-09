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
import os

# reload modules
for module in [density, plots, kernel, losses, models, sampler, stats]:
    importlib.reload(module)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    t = max(t, 0.2)
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
    mass_less_zero = np.sum(f[x < 0]) * (x[1] - x[0])
    plt.plot(x, f, label=f't={t}, mass<0: {mass_less_zero:.2f}')

plt.title(r'Gaussian Interpolation $0.1 \mathcal{N}(-5t, 1) + 0.9 \mathcal{N}(5t, 1)$')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for t in t_values:
    f = dilation_interpolation(target_density, t)
    f_vals = f(x)
    mass_less_zero = np.sum(f_vals[x < 0]) * (x[1] - x[0])
    plt.plot(x, f_vals, label=f't={t}, mass<0: {mass_less_zero:.2f}')

plt.title(r'Dilation Interpolation $\pi_t(x) = \pi(x/t)/t$')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for t in t_values:
    f = geometric_interpolation(standard_gaussian, target_density, t)
    f_vals = f(x)
    mass_less_zero = np.sum(f_vals[x < 0]) * (x[1] - x[0])
    plt.plot(x, f_vals, label=f't={t}, mass<0: {mass_less_zero:.2f}')

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
importlib.reload(sampler)
for (step_size, max_steps) in [(0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000)]:
    for (λ, lambda_name) in [(λ1, 't')]:
        # train initial score model
        mlp = models.MLP(d=1)
        score_model = models.ResNet(mlp)
        optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
        prior_score_values = prior_score(prior_sample)
        print("Training initial NN. Losses:")
        for i in range(71):
            if i % 10 == 0:
                print(losses.explicit_score_matching_loss(score_model, prior_sample, prior_score_values))
            loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
            optimizer.update(grads)

        print(f"Lambda(t) = {lambda_name}")
        print(f"Step size={step_size}, Max steps={max_steps}, t_end={step_size * max_steps}")
        t_end = step_size * max_steps

        annealed_score = lambda t,x : geometric_mean_score(λ(t, t_end), x, prior_score, target_score)

        logger = sampler.Logger()
        sampler.SBTMSampler(prior_sample, annealed_score, step_size, max_steps, logger, score_model, losses.implicit_score_matching_loss, optimizer, gd_stopping_criterion=sampler.FixedNumBatches(20)).sample()

        # count < 0
        particles = logger.get_trajectory('particles')
        print(f"Step size={step_size}, Max steps={max_steps}")
        negative_counts = []
        for i in range(0, max_steps, max_steps//4):
            p = particles[i][:,0]    
            negative_count = np.sum(p < 0)
            negative_counts.append(negative_count)
            print(f"i={i}, Count of particles < 0: {negative_count}")

        # plot KDE
        particles = logger.get_trajectory('particles')
        time_steps = np.linspace(0, max_steps-1, 11)
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

        plot_dir = os.path.expanduser('~/SBTM-sampling/plots/annealing/sbtm/geometric_mean')
        data_dir = os.path.expanduser('~/SBTM-sampling/data/annealing/sbtm/geometric_mean')
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        plt.savefig(os.path.join(plot_dir, f'kde_particles_{step_size}_{max_steps}_lambda_{lambda_name}.png'))
        plt.show()

        # Plot KL divergence over time
        kl_divergences_annealed = []
        kl_divergences_target = []
        times = logger.get_trajectory('time')
        ts = []
        for i in list(range(0, max_steps, max_steps//10)) + [max_steps-1]:
            t = times[i]
            ts.append(t)
            particles = logger.get_trajectory('particles')[i]
            
            # KL divergence with annealed interpolation
            annealed_density = geometric_interpolation(standard_gaussian, target_density, λ(t, t_end))
            kl_div_annealed = stats.kl_divergence(particles, annealed_density)
            
            # KL divergence with target density
            kl_div_target = stats.kl_divergence(particles, target_density)
            
            kl_divergences_annealed.append(kl_div_annealed)
            kl_divergences_target.append(kl_div_target)
            print(f"i={i}, t={t:.2f}, KL divergence (annealed): {kl_div_annealed:.4f}, KL divergence (target): {kl_div_target:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(ts, kl_divergences_annealed, marker='o', label=r'KL$(f || \pi_t)$')
        plt.plot(ts, kl_divergences_target, marker='x', label=r'KL$(f || \pi)$')
        plt.yscale('log')
        plt.title('KL Divergence over Time')
        plt.xlabel('Time')
        plt.ylabel('KL Divergence')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'kl_divergence_{step_size}_{max_steps}_lambda_{lambda_name}.png'))
        plt.show()

        # plot the fisher divergence and time derivative of KL divergence
        steps_to_plot = max_steps
        fig, ax = plt.subplots(figsize=(10, 6))
        smoothing = 0.5

        # Compute KL divergences
        yscale = 'log'
        kl_divs = stats.compute_kl_divergences(logger.get_trajectory('particles'), target_density)
        kl_divs = jnp.array(kl_divs)
        kl_div_time_derivative = -jnp.diff(kl_divs) / step_size
        kl_div_time_derivative = jnp.clip(kl_div_time_derivative, a_min=1e-5)
        plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative[:steps_to_plot], smoothing), label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$', marker='o', markersize=3, yscale=yscale)

        # Compute Fisher divergences
        particles = logger.get_trajectory('particles')
        scores = logger.get_trajectory('score')
        times = logger.get_trajectory('time')
        fisher_divs = []
        for t, particles_i, scores_i in tqdm(list(zip(times, particles, scores)), desc="Computing Fisher divergence"):
            annealed_score_values = annealed_score(t, particles_i)
            value = jnp.mean(jax.vmap(stats.square_norm_diff)(scores_i, annealed_score_values))
            fisher_divs.append(value)
        plots.plot_quantity_over_time(ax, stats.ema(fisher_divs[:steps_to_plot], smoothing), label=r'annealed: $\frac{1}{n}\sum_{i=1}^n\|\nabla \log \pi_t(X_i) - s(X_i)\|^2$', yscale=yscale)

        fisher_divs = jnp.array(stats.compute_fisher_divergences(particles, scores, target_score))
        plots.plot_quantity_over_time(ax, stats.ema(fisher_divs[:steps_to_plot], smoothing), label=r'target: $\frac{1}{n}\sum_{i=1}^n\|\nabla \log \pi(X_i) - s(X_i)\|^2$', yscale=yscale)

        ax.set_title("KL divergence decay rate and Fisher divergence")
        ax.set_xlabel('Time')
        ax.set_ylabel('Divergence')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(plot_dir, f'fisher_kl_divergence_{step_size}_{max_steps}_lambda_{lambda_name}.png'))
        plt.show()

        # save
        log_data = {
            'logs': logger.logs,
            'hyperparameters': logger.hyperparameters
        }
        with open(os.path.join(data_dir, f'logger_{step_size}_{max_steps}_lambda_{lambda_name}.pkl'), 'wb') as f:
            pickle.dump(log_data, f)

#%%
