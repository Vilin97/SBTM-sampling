#%%
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler
from flax import nnx
import optax
# import os
# os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'
import matplotlib.pyplot as plt

#%%
# reload modules
for module in [density, plots, kernel, losses, models, sampler]:
    importlib.reload(module)

#%%
# set up
step_size = 0.1
max_steps = 50
num_particles = 5000
key = jrandom.key(42)

prior_params = {'mean': jnp.array([0.]), 'covariance': jnp.array([[10.]])}
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))
prior_score = density.Density(density.gaussian_pdf, prior_params).score

target_params = {'mean': jnp.array([0.]), 'covariance': jnp.array([[1.]])}
target_density_obj = density.Density(density.gaussian_pdf, target_params)
target_score = target_density_obj.score

#%%
# sample with sde
sde_logger = sampler.Logger()
sde_sampler = sampler.SDESampler(prior_sample, target_score, step_size, max_steps, sde_logger)
sde_sample = sde_sampler.sample()
fig, ax = plots.plot_distributions(prior_sample, sde_sample, target_density_obj)
ax.set_title('SDE')
ax.set_xlim(-10, 10)
fig.show()

plots.plot_kl_divergence(sde_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# sample with svgd
svgd_kernel = kernel.Kernel(kernel.rbf_kernel)
svgd_logger = sampler.Logger()
svgd_sampler = sampler.SVGDSampler(prior_sample, target_score, step_size, max_steps, svgd_logger, svgd_kernel)
svgd_sample = svgd_sampler.sample()
fig, ax = plots.plot_distributions(prior_sample, svgd_sample, target_density_obj)
ax.set_title('SVGD')
ax.set_xlim(-10, 10)
fig.show()

plots.plot_kl_divergence(svgd_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# train initial score model
importlib.reload(losses)
mlp = models.MLP(d=1, hidden_units=[128])
score_model = models.ResNet(mlp)
optimizer = nnx.Optimizer(score_model, optax.adamw(0.001, 0.9))
for i in range(50):
    if i % 10 == 0:
        print(losses.explicit_score_matching_loss(score_model, prior_score, prior_sample))
    loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_score, prior_sample)
    optimizer.update(grads)
print(losses.explicit_score_matching_loss(score_model, prior_score, prior_sample))
#%%
# sample with sbtm
sbtm_logger = sampler.Logger()
loss = losses.implicit_score_matching_loss
sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer)
sbtm_sample = sbtm_sampler.sample()
fig, ax = plots.plot_distributions(prior_sample, sbtm_sample, target_density_obj)
ax.set_title('SBTM')
ax.set_xlim(-10, 10)
fig.show()

plots.plot_kl_divergence(sbtm_logger.get_trajectory('particles'), target_density_obj.density)

# %%
loss_values = [loss_value for log in sbtm_logger.logs for loss_value in log['loss_values']]
batch_loss_values = [loss_value for log in sbtm_logger.logs for loss_value in log['batch_loss_values']]

def exponential_moving_average(data, smoothing):
    ema = []
    ema_current = data[0]
    for value in data:
        ema_current = (1 - smoothing) * value + smoothing * ema_current
        ema.append(ema_current)
    return ema

ema_losses = exponential_moving_average(loss_values, smoothing=0.4)
ema_batch_losses = exponential_moving_average(batch_loss_values, smoothing=0.95)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_values, label='Losses')
# plt.plot(ema_losses, label='Exponential Moving Average', color='red')
plt.xlabel('Iteration')
plt.ylabel(r'$\frac{1}{n} \sum_{i} ||s(x_{i})||^2 + 2 \nabla \cdot s(x_{i})$')
plt.title(r'Implicit loss through iterations')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(batch_loss_values, label='Batch Losses')
plt.plot(ema_batch_losses, label='Exponential Moving Average', color='red')
plt.xlabel('Iteration')
plt.ylabel(r'Batch Loss')
plt.title(r'Batch loss through iterations')
plt.legend()

plt.tight_layout()
plt.show()

#%% 
# compute fisher divergence
def fisher_divergence(score_values_1, score_values_2):
    return jnp.sum(jnp.square(score_values_1 - score_values_2))
    
fisher_divs = []
for log in sbtm_logger.logs:
    particles = log['particles']
    score = log['score']
    value = jnp.mean(jax.vmap(fisher_divergence)(score, target_score(particles)))
    fisher_divs.append(value)
    
plt.plot(fisher_divs, label='Fisher Divergence')
plt.yscale('log')
plt.title('Fisher Divergence Estimate')
plt.xlabel('Step')
plt.ylabel(r'$\frac{1}{n} \sum_{i} ||s(x_{i}) - \nabla \log \pi(x_{i})||^2$')
plt.legend()
plt.show()

# %%
import seaborn as sns
from scipy.stats import gaussian_kde

def kde(x_values, logs):
    iterations = [log['particles'][:, 0] for log in logs]
    density_values = []
    for particles in iterations:
        kde = gaussian_kde(particles)
        density_values.append(kde(x_values))
    return density_values

def plot_density_evolution(x_values, density_values, title, trajectories):
    assert len(x_values) == len(density_values[0])
    assert len(density_values) == len(trajectories[0])
    xmin, xmax = x_values[0], x_values[-1]
    num_x_values = len(x_values)
    num_iterations = len(density_values)
    
    sns.heatmap(jnp.array(density_values)[::-1,:])
    plt.xticks(ticks=jnp.linspace(0, num_x_values, 9), labels=[f'{int(x)}' for x in jnp.linspace(xmin, xmax, 9)], rotation=0)
    plt.yticks(ticks=jnp.linspace(0, num_iterations, 9), labels=[f'{int(x)}' for x in jnp.linspace(num_iterations, 0, 9)], rotation=0)
    plt.ylabel('Iteration')
    plt.title(title)
    
    for trajectory in trajectories:
        trajectory_mapped = [jnp.argmin(jnp.abs(x_values - value)) for value in trajectory[::-1]]
        plt.plot(trajectory_mapped, jnp.linspace(0, len(density_values), len(trajectory)), color='white', marker='o', markersize=2)
    plt.show()

def visualize_trajectories(logs, title, particle_idxs = [0,1,2,3,4]):
    x_values = jnp.linspace(-10, 10, 200)
    sde_kde = kde(x_values, logs)
    trajectories = [[log['particles'][i, 0] for log in logs] for i in particle_idxs]
    plot_density_evolution(x_values, sde_kde, title, trajectories)

visualize_trajectories(sde_logger.logs, 'SDE')
visualize_trajectories(sbtm_logger.logs, 'SBTM')
visualize_trajectories(svgd_logger.logs, 'SVGD')

# %%