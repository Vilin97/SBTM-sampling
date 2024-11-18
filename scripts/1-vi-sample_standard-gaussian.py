#%%
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler, stats
from flax import nnx
import optax
# import os
# os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
# reload modules
for module in [density, plots, kernel, losses, models, sampler]:
    importlib.reload(module)

#%%
# set up
step_size = 0.1
max_steps = 30
num_particles = 1000
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
#%%
fig, ax = plots.plot_distributions(prior_sample, sde_sample, target_density_obj)
ax.set_xlim(-10, 10)
ax.set_title('SDE')
fig.show()    
plots.visualize_trajectories(sde_logger.get_trajectory('particles'), title='SDE')
plots.plot_kl_divergence(sde_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# sample with svgd
svgd_kernel = kernel.Kernel(kernel.rbf_kernel)
svgd_logger = sampler.Logger()
svgd_sampler = sampler.SVGDSampler(prior_sample, target_score, step_size, max_steps, svgd_logger, svgd_kernel)
svgd_sample = svgd_sampler.sample()
fig, ax = plots.plot_distributions(prior_sample, svgd_sample, target_density_obj)
ax.set_xlim(-10, 10)
ax.set_title('SVGD')
fig.show()

plots.visualize_trajectories(svgd_logger.get_trajectory('particles'), title='SVGD')
plots.plot_kl_divergence(svgd_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# train initial score model
mlp = models.MLP(d=1)
score_model = models.ResNet(mlp)
optimizer = nnx.Optimizer(score_model, optax.adamw(0.001, 0.9))
for i in tqdm(range(100)):
    if i % 10 == 0:
        print(losses.explicit_score_matching_loss(score_model, prior_sample, prior_score(prior_sample)))
    loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score(prior_sample))
    optimizer.update(grads)
print(losses.explicit_score_matching_loss(score_model, prior_sample, prior_score(prior_sample)))
#%%
# sample with sbtm
sbtm_logger = sampler.Logger()
loss = losses.implicit_score_matching_loss
sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer, gd_stopping_criterion=sampler.FixedNumBatches(10), mini_batch_size=num_particles)
# sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer, gd_stopping_criterion=sampler.AbsoluteLossChange(0.01), mini_batch_size=num_particles)
# sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer, gd_stopping_criterion=sampler.AdaptiveNumBatches(step_size, losses.approx_explicit_score_matching_loss, k=1), mini_batch_size=num_particles)
sbtm_sample = sbtm_sampler.sample()

#%%
fig, ax = plots.plot_distributions(prior_sample, sbtm_sample, target_density_obj)
ax.set_title('SBTM')
ax.set_xlim(-10, 10)
fig.show()

plots.visualize_trajectories(sbtm_logger.get_trajectory('particles'), title='SBTM')
# plots.plot_kl_divergence(sbtm_logger.get_trajectory('particles'), target_density_obj.density)
plots.plot_fisher_divergence(sbtm_logger.get_trajectory('particles'), target_score, sbtm_logger.get_trajectory('score'))

#%%
# plot the fisher divergence and time derivative of KL divergence
steps_to_plot = 100
fig, ax = plt.subplots(figsize=(6, 6))
smoothing = 0.5

for (logger, name) in zip([sde_logger, sbtm_logger, svgd_logger], ['SDE', 'SBTM', 'SVGD']):
    try:
        kde_kl_divs = stats.compute_kl_divergences(logger.get_trajectory('particles'), target_density_obj.density)
        kde_kl_divs = jnp.array(kde_kl_divs)
        kl_div_time_derivative = -jnp.diff(kde_kl_divs) / step_size
        kl_div_time_derivative = jnp.clip(kl_div_time_derivative, a_min=1e-5)
        plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative[:steps_to_plot], smoothing), label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, {name}', marker='o', markersize=3)
    except:
        print(f"Could not compute KL divergence for {name}")

particles = sbtm_logger.get_trajectory('particles')
sbtm_scores = sbtm_logger.get_trajectory('score')
sbtm_fisher_divs = jnp.array(stats.compute_fisher_divergences(particles, sbtm_scores, target_score))
plots.plot_quantity_over_time(ax, stats.ema(sbtm_fisher_divs[:steps_to_plot], smoothing), label=r'NN: $\frac{1}{n}\sum_{i=1}^n\|\nabla \log \pi_t(X_i) - s(X_i)\|^2$')

kde_scores = [stats.compute_score(sample_f) for sample_f in particles]
kde_fisher_divs = jnp.array(stats.compute_fisher_divergences(particles, kde_scores, target_score))
plots.plot_quantity_over_time(ax, stats.ema(kde_fisher_divs[:steps_to_plot], smoothing), label=r'KDE: $\frac{1}{n}\sum_{i=1}^n\|\nabla \log \pi_t(X_i) - \nabla \frac{1}{n}\sum_{j=1}^n \phi_\varepsilon(X_i-X_j)\|^2$', max_time=max_steps*step_size)

ax.set_yscale('log')
ax.set_title("KL divergence decay rate")
fig.show()