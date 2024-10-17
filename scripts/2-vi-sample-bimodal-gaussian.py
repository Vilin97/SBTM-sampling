#%%
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler
from flax import nnx
import optax
import os
os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'
import matplotlib.pyplot as plt

#%%
# reload modules
for module in [density, plots, kernel, losses, models, sampler]:
    importlib.reload(module)

#%%
# set up
step_size = 0.2
max_steps = 200
num_particles = 1000
key = jrandom.key(42)

prior_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[10.]])}
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))
prior_score = density.Density(density.gaussian_pdf, prior_params).score

target_params = {'mean': [jnp.array([-5]), jnp.array([5])], 'covariance': [jnp.array([[1.]]), jnp.array([[1.]])], 'weights': jnp.array([1/10, 1-1/10])}
target_density_obj = density.Density(density.gaussian_mixture_pdf, target_params)
target_score = target_density_obj.score

#%%
# sample with sde
sde_logger = sampler.Logger()
sde_sampler = sampler.SDESampler(prior_sample, target_score, step_size, max_steps, sde_logger)
sde_sample = sde_sampler.sample()
fig, ax = plots.plot_distributions(prior_sample, sde_sample, target_density_obj)
ax.set_xlim(-10, 10)
ax.set_title('SDE')
fig.show()    

plots.visualize_trajectories(sde_logger.get_trajectory('particles'), 'SDE')
plots.plot_kl_divergence(sde_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# sample with svgd
svgd_kernel = kernel.Kernel(kernel.rbf_kernel)
svgd_logger = sampler.Logger()
svgd_sampler = sampler.SVGDSampler(prior_sample, target_score, step_size, max_steps, svgd_logger, svgd_kernel)
svgd_sample = svgd_sampler.sample()
fig, ax = plots.plot_distributions(prior_sample, svgd_sample, target_density_obj)
ax.set_xlim(-10, 10)
fig.show()

plots.visualize_trajectories(svgd_logger.get_trajectory('particles'), 'SVGD')
plots.plot_kl_divergence(svgd_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# train initial score model
mlp = models.MLP(d=1)
score_model = models.ResNet(mlp)
optimizer = nnx.Optimizer(score_model, optax.adamw(0.001, 0.9))
for i in range(100):
    if i % 10 == 0:
        print(losses.explicit_score_matching_loss(score_model, prior_score, prior_sample))
    loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_score, prior_sample)
    optimizer.update(grads)
print(losses.explicit_score_matching_loss(score_model, prior_score, prior_sample))
#%%
# sample with sbtm
sbtm_logger = sampler.Logger()
loss = losses.implicit_score_matching_loss
sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, 50, sbtm_logger, score_model, loss, optimizer)
sbtm_sample = sbtm_sampler.sample()
fig, ax = plots.plot_distributions(prior_sample, sbtm_sample, target_density_obj)
ax.set_title('SBTM')
ax.set_xlim(-10, 10)
fig.show()

plots.visualize_trajectories(sbtm_logger.get_trajectory('particles'), 'SBTM')
plots.plot_kl_divergence(sbtm_logger.get_trajectory('particles'), target_density_obj.density)
plots.plot_fisher_divergence(sbtm_logger.get_trajectory('particles'), sbtm_logger.get_trajectory('score'), target_score)

#%%
# Rollout: sample from noised target, use the sample as initial condition for less noised target, etc
targets = []
noise_levels = [15,9,4,2,1,0]
for i in noise_levels:
    params = {'mean': [jnp.array([-5]), jnp.array([5])], 'covariance': [jnp.array([[1.+i]]), jnp.array([[1.+i]])], 'weights': jnp.array([1/3, 2/3])}
    target_i = density.Density(density.gaussian_mixture_pdf, params)
    targets.append(target_i)
    x_values = jnp.linspace(-10, 10, 1000)
    plt.plot(x_values, target_i.density(x_values.reshape(1000,1)), label=f'noise = {i}')
plt.legend()
plt.show()

score_model = models.ResNet(models.MLP(d=1, hidden_units=[128,128]))
loss = losses.implicit_score_matching_loss
optimizer = nnx.Optimizer(score_model, optax.adamw(0.001, 0.9))

logger = sampler.Logger()
prior_sample_i = prior_sample.copy()
for (i, target) in enumerate(targets):
    sampler_obj = sampler.SBTMSampler(prior_sample_i, target.score, step_size, max_steps, logger, score_model, loss, optimizer)
    # sampler_obj = sampler.SDESampler(prior_sample_i, target.score, step_size, max_steps, logger)
    sample = sampler_obj.sample()
    print(f'Noise level: {noise_levels[i]}, covariance: {target.params["covariance"][0][0]}')
    fig, ax =plots.plot_distributions(prior_sample_i, sample, target)
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 0.3)
    ax.set_title(f'Noise level: {noise_levels[i]}')
    fig.show()
    prior_sample_i = sample
    
# %%
# plot trajectories
plots.visualize_trajectories(logger.get_trajectory('particles')[::20], 'SBTM', particle_idxs=range(1))

# %%
# plot kl divergence
plots.plot_kl_divergence(logger.get_trajectory('particles')[::10], target_density_obj.density)

# %%
# plot losses
loss_values = [loss_value for log in logger.logs for loss_value in log['loss_values']]
batch_loss_values = [loss_value for log in logger.logs for loss_value in log['batch_loss_values']]
plots.plot_losses(loss_values, batch_loss_values)

#%% 
# plot fisher divergence
plots.plot_fisher_divergence(logger.get_trajectory('particles')[::10], logger.get_trajectory('score')[::10], target_score)

# %%
