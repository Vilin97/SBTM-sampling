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
import jax
from flax.training import common_utils
for module in [density, plots, kernel, losses, models, sampler, stats, distribution]:
    importlib.reload(module)

# Set the memory fraction for JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.45'
# Set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
import jax.numpy as jnp

def train_step(particles, score_model, optimizer, mini_batch_size=200, debug=False, prng_key=0):
    loss = losses.implicit_score_matching_loss

    # Shuffle and batch the data
    particles = jax.random.permutation(jrandom.PRNGKey(prng_key), particles)
    num_batches = len(particles) // mini_batch_size
    batches = jnp.array_split(particles, num_batches)

    # One epoch
    for batch in batches:
        loss_value = sampler.opt_step(score_model, optimizer, loss, batch)
    if debug:
        print(f"Loss: {loss_value}")
#%%
def K(t):
    return 1 - jnp.exp(-2*t)

t0 = 0.1
key = jrandom.key(47)

prior_dist = distribution.Gaussian(jnp.array([0.]), jnp.array([[K(t0)]]))
prior_density = prior_dist.density
prior_score = prior_dist.score

target_dist = distribution.Gaussian(jnp.array([0.]), jnp.array([[1.]])) 

# #%%
# example_name = 'analytic'
# annealing_name = 'non-annealed'
# step_size = 0.01
# max_steps = 500
# method_name = 'sbtm'

# num_particles = 1000
# num_particles_str = '' if num_particles==1000 else '_particles_10000'

# # load particles
# data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
# path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}.pkl')
# with open(path, 'rb') as f:
#     log_data = pickle.load(f)
# particles = [log['particles'] for log in log_data['logs']]

# # train initial NN
# prior_sample = particles[0]
# prior_score_values = prior_score(prior_sample)
# score_model = models.ResNet(models.MLP(d=prior_sample.shape[1]))
# optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
# for i in range(1001):
#     loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
#     optimizer.update(grads)
#     if i % 100 == 0:
#         print(f"Loss: {loss_value}")

# #%%
# plots.plot_distributions(prior_sample, particles[-1], target_dist.density)

# # Plot the score model and prior score
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot prior score
# x = jnp.linspace(-3, 3, 1000).reshape(-1, 1)
# prior_score_values = prior_score(x)
# ax.plot(x, prior_score_values, label='Prior Score', color='blue')

# # Plot score model
# score_model_values = score_model(x)
# ax.plot(x, score_model_values, label='Score Model', color='green')

# ax.set_title('Score Model vs Prior Score')
# ax.legend()
# fig.show()

# #%%
# # compare the sbtm score values in simulation to the post-trained ones
# # train initial NN
# prior_sample = particles[0]
# prior_score_values = prior_score(prior_sample)
# score_model = models.ResNet(models.MLP(d=prior_sample.shape[1]))
# optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
# for i in range(1001):
#     loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
#     optimizer.update(grads)
#     if i % 100 == 0:
#         print(f"Loss: {loss_value}")

# # train NN
# score_values = []

# for particles_i in tqdm(particles, desc='Training'):
#     train_step(particles_i, score_model, optimizer)
#     score_values.append(score_model(particles_i))

#%%
example_name = 'analytic'
annealing_name = 'non-annealed'
# step_size = 0.002
# max_steps = 2500
# num_particles = 10000
step_size = 0.01
max_steps = 500
num_particles = 1000

num_particles_str = '' if num_particles==1000 else '_particles_10000'

for method_name in ['sbtm', 'sde']:
    # load particles
    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}.pkl')
    with open(path, 'rb') as f:
        log_data = pickle.load(f)
    particles = [log['particles'] for log in log_data['logs']]
    

    # train initial NN
    prior_sample = particles[0]
    prior_score_values = prior_score(prior_sample)
    score_model = models.ResNet(models.MLP(d=prior_sample.shape[1]))
    optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
    for i in range(1001):
        loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
        optimizer.update(grads)
        if i % 100 == 0:
            print(f"Loss: {loss_value}")

    # train NN
    score_values = []
    # TODO: for some reason, the NN is overfitting, badly
    for (i, particles_i) in tqdm(list(enumerate(particles)), desc=f'Training NN, {method_name}'):
        train_step(particles_i, score_model, optimizer, prng_key=i, mini_batch_size=num_particles)
        score_values.append(score_model(particles_i))

    # Save score values
    score_values_path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}_score.pkl')
    with open(score_values_path, 'wb') as f:
        pickle.dump(score_values, f)

    #### plot ####
    target_dist = distribution.Gaussian(jnp.array([0.]), jnp.array([[1.]]))
    smoothing = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    steps_to_plot = max_steps//2
    T = step_size * max_steps

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
    plots.plot_quantity_over_time(ax, stats.ema(analytic_kl_div_time_derivative, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, Analytic', marker='o', markersize=3, yscale='log', max_time=T, color='red')


    ax.set_yscale('log')
    ax.set_title(f"{example_name} $\Delta t={step_size}$, $T={T}$")
    fig.show()

    save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/entropy_dissipation/{method_name}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}.png'))

# %%
example_name = 'analytic'
annealing_name = 'non-annealed'
method_name = 'sbtm'
step_size = 0.01
max_steps = 500
num_particles = 1000
# step_size = 0.002
# max_steps = 2500
# num_particles = 10000
num_particles_str = '' if num_particles==1000 else '_particles_10000'

data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
score_values_path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}_score.pkl')
with open(score_values_path, 'rb') as f:
    new_score_values = pickle.load(f)
    
score_values_path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}.pkl')
with open(score_values_path, 'rb') as f:
    sbtm_logs = pickle.load(f)
#%%
old_score_values = [log['score'] for log in sbtm_logs['logs']]
particles = [log['particles'] for log in sbtm_logs['logs']]
#%%
true_score_values = []
ts = jnp.linspace(0.1, 0.1 + max_steps * step_size, max_steps)
for (particles_i, t) in tqdm(list(zip(particles, ts)), desc='Computing True Score Values'):
    dist_t = distribution.Gaussian(jnp.array([0.]), jnp.array([[K(t)]]))
    true_score_values.append(dist_t.score(particles_i))
#%%
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(jnp.mean(jnp.array(old_score_values), axis=1), label='Old Score Values')
ax.plot(jnp.mean(jnp.array(new_score_values), axis=1), label='New Score Values')
ax.plot(jnp.mean(jnp.array(true_score_values), axis=1), label='True Score Values')
ax.legend()
fig.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(jnp.mean(jnp.array(old_score_values) - jnp.array(true_score_values), axis=1), label='old - true')
ax.plot(jnp.mean(jnp.array(new_score_values) - jnp.array(true_score_values), axis=1), label='new - true')
ax.legend()
fig.show()

#%%
# for t_idx in [0, 100, 500, 1000, 1500, 2000]:
for t_idx in [0, 50, 100, 200, 400]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_indices = jnp.argsort(particles[t_idx], axis=0).flatten()
    sorted_particles = particles[t_idx][sorted_indices]
    sorted_true_scores = true_score_values[t_idx][sorted_indices]
    sorted_old_scores = old_score_values[t_idx][sorted_indices]
    sorted_new_scores = new_score_values[t_idx][sorted_indices]

    ax.plot(sorted_particles, sorted_true_scores, label='True Score')
    ax.plot(sorted_particles, sorted_old_scores, label='Old Score')
    ax.plot(sorted_particles, sorted_new_scores, label='New Score')
    ax.legend()
    ax.set_title(f"t={t_idx}")
    fig.show()

#%%
smoothing = 0.5
steps_to_plot = max_steps//2
T = step_size * steps_to_plot

fig, ax = plt.subplots(figsize=(10, 6))

# entropy dissipation
kl_div_time_derivative = -stats.time_derivative(stats.compute_kl_divergences(particles, target_dist.log_density), step_size)
kl_div_time_derivative = jnp.where(jnp.isnan(kl_div_time_derivative), jnp.nanmax(kl_div_time_derivative), kl_div_time_derivative)
kl_div_time_derivative = jnp.clip(kl_div_time_derivative, a_min=1e-5, a_max=1e4)
plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, {method_name}', marker='o', markersize=3, max_time=T)


# # analytic KL dissipation
analytic_kl_divs = []
for t in tqdm(jnp.linspace(0.1, 0.1 + max_steps * step_size, max_steps), desc="Computing analytic KL divergences"):
    K_t = K(t)
    analytic_cov = jnp.array([[K_t]])
    analytic_kl_div = stats.relative_entropy_gaussians(jnp.array([0.]), analytic_cov, jnp.array([0.]), jnp.array([[1.]]))
    analytic_kl_divs.append(analytic_kl_div)
analytic_kl_div_time_derivative = -jnp.diff(jnp.array(analytic_kl_divs)) / step_size
analytic_kl_div_time_derivative = jnp.clip(analytic_kl_div_time_derivative, a_min=1e-5, a_max = 1e4)
plots.plot_quantity_over_time(ax, stats.ema(analytic_kl_div_time_derivative, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, Analytic', marker='o', markersize=3, yscale='log', max_time=T, color='red')

# relative fisher info
sbtm_fisher_divs = jnp.array(stats.compute_fisher_divergences(particles, true_score_values, target_dist.score))
plots.plot_quantity_over_time(ax, stats.ema(sbtm_fisher_divs, smoothing)[:steps_to_plot], label=rf'true score', max_time=T)

sbtm_fisher_divs = jnp.array(stats.compute_fisher_divergences(particles, old_score_values, target_dist.score))
plots.plot_quantity_over_time(ax, stats.ema(sbtm_fisher_divs, smoothing)[:steps_to_plot], label=rf'old score', max_time=T)

sbtm_fisher_divs = jnp.array(stats.compute_fisher_divergences(particles, new_score_values, target_dist.score))
plots.plot_quantity_over_time(ax, stats.ema(sbtm_fisher_divs, smoothing)[:steps_to_plot], label=rf'new score', max_time=T)

ax.set_yscale('log')
fig.show()

# %%
