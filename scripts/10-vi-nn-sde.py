#%%
import os
import pickle 
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
import numpy as np
from flax import nnx
import optax

from sbtm import density, plots, kernel, losses, models, sampler, stats, distribution
for module in [density, plots, kernel, losses, models, sampler, stats, distribution]:
    importlib.reload(module)

# Set the memory fraction for JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.45'
# Set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
def train_step(particles, score_model, optimizer, mini_batch_size=100, debug=False):
    batch_loss_values = []
    loss_values = []
    num_particles = particles.shape[0]
    num_batches = num_particles // mini_batch_size
    loss = losses.implicit_score_matching_loss

    # one epoch
    loss_values.append(loss(score_model, particles))
    for i in range(num_batches):
        batch_start = i * mini_batch_size
        batch_end = batch_start + mini_batch_size
        batch = particles[batch_start:batch_end, :]
        loss_value = sampler.opt_step(score_model, optimizer, loss, batch)
        batch_loss_values.append(loss_value)
    if debug:
        print(f"Loss: {loss_values[-1]}")
    
    loss_values.append(loss(score_model, particles))
    return loss_values, batch_loss_values

#%%
def K(t):
    return 1 - jnp.exp(-2*t)

t0 = 0.1
key = jrandom.key(47)

prior_dist = distribution.Gaussian(jnp.array([0.]), jnp.array([[K(t0)]]))
prior_density = prior_dist.density
prior_score = prior_dist.score

#%%
example_name = 'analytic'
annealing_name = 'non-annealed'
step_size = 0.002
max_steps = 2500
for method_name in ['sde', 'sbtm']:
    # load particles
    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}_particles_10000.pkl')
    with open(path, 'rb') as f:
        log_data = pickle.load(f)
    particles = [log['particles'] for log in log_data['logs']]

    # train initial NN
    prior_sample = particles[0]
    prior_score_values = prior_score(prior_sample)
    score_model = models.ResNet(models.MLP(d=prior_sample.shape[1]))
    optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
    for i in range(101):
        loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
        optimizer.update(grads)
        if i % 10 == 0:
            print(f"Loss: {loss_value}")

    # train NN
    score_values = []

    for particles_i in tqdm(particles, desc='Training'):
        train_step(particles_i, score_model, optimizer)
        score_values.append(score_model(particles[i]))

    # Save score values
    score_values_path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}_particles_10000_score.pkl')
    with open(score_values_path, 'wb') as f:
        pickle.dump(score_values, f)

    #### plot ####
    target_dist = distribution.Gaussian(jnp.array([0.]), jnp.array([[1.]]))
    smoothing = 0.5
    T = step_size * max_steps

    fig, ax = plt.subplots(figsize=(10, 6))
    steps_to_plot = max_steps//2

    # entropy dissipation
    kl_div_time_derivative = -stats.time_derivative(stats.compute_kl_divergences(particles, target_dist.log_density), step_size)
    kl_div_time_derivative = jnp.where(jnp.isnan(kl_div_time_derivative), jnp.nanmax(kl_div_time_derivative), kl_div_time_derivative)
    kl_div_time_derivative = jnp.clip(kl_div_time_derivative, a_min=1e-5, a_max=1e4)
    plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, {method_name}', marker='o', markersize=3, max_time=T)

    # relative fisher info
    sbtm_fisher_divs = jnp.array(stats.compute_fisher_divergences(particles, score_values, target_dist.score))
    plots.plot_quantity_over_time(ax, stats.ema(sbtm_fisher_divs, smoothing)[:steps_to_plot], label=rf'$\frac{{1}}{{n}}\sum_{{i=1}}^n\|\nabla \log \pi_t(X_i) - s(X_i)\|^2$, {method_name}', max_time=T)

    # analytic KL dissipation
    analytic_kl_divs = []
    for t in tqdm(jnp.linspace(0.1, 0.1 + max_steps * step_size, max_steps), desc="Computing analytic KL divergences"):
        K_t = K(t)
        analytic_cov = jnp.array([[K_t]])
        analytic_kl_div = stats.relative_entropy_gaussians(jnp.array([0.]), analytic_cov, jnp.array([0.]), jnp.array([[1.]]))
        analytic_kl_divs.append(analytic_kl_div)

    analytic_kl_div_time_derivative = -jnp.diff(jnp.array(analytic_kl_divs)) / step_size
    analytic_kl_div_time_derivative = jnp.clip(analytic_kl_div_time_derivative, a_min=1e-5, a_max = 1e4)
    plots.plot_quantity_over_time(ax, stats.ema(analytic_kl_div_time_derivative, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, Analytic', marker='o', markersize=3, yscale='log', max_time=T)


    ax.set_yscale('log')
    ax.set_title(f"{example_name} $\Delta t={step_size}$, $T={T}$")
    fig.show()

    save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/entropy_dissipation/{method_name}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}_particles_10000.png'))

