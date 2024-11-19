# Compare the analytic solution of the following FP equation with the SDE and SBTM solutions:
# f(t) = (2\pi K)^{-n/2}\exp\left(-\frac{|x|^2}{2K}\right), \; K=1-\exp(-2t)

#%%
import importlib
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler, stats
from flax import nnx
import optax
import os
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from scipy import integrate
# reload modules
for module in [density, plots, kernel, losses, models, sampler, stats]:
    importlib.reload(module)

#%%
# set up
def K(t):
    return 1 - jnp.exp(-2*t)
def f(t, x):
    return (2*jnp.pi*K(t))**(-0.5)*jnp.exp(-jnp.sum(x**2)/(2*K(t)))

step_size = 0.01
max_steps = 500
num_particles = 1000
key = jrandom.key(42)
t0 = 0.1

prior_params = {'mean': jnp.array([0.]), 'covariance': jnp.array([[K(t0)]])}
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))
prior_score = density.Density(density.gaussian_pdf, prior_params).score

target_params = {'mean': jnp.array([0.]), 'covariance': jnp.array([[1.]])}
target_density_obj = density.Density(density.gaussian_pdf, target_params)
target_score = target_density_obj.score
plt.plot(jnp.linspace(-10, 10, 1000), density.Density(density.gaussian_pdf, prior_params).density(jnp.reshape(jnp.linspace(-10, 10, 1000), (1000,1))), label='Prior Density')
plt.plot(jnp.linspace(-10, 10, 1000), target_density_obj.density(jnp.reshape(jnp.linspace(-10, 10, 1000), (1000,1))), label='Target Density')
plt.legend()

#%%
# sample with sde
sde_logger = sampler.Logger()
sde_sampler = sampler.SDESampler(prior_sample, target_score, step_size, max_steps, sde_logger)
sde_sample = sde_sampler.sample()
#%%
fig, ax = plots.plot_distributions(prior_sample, sde_sample, target_density_obj)
ax.set_xlim(-10, 10)
ax.set_title(fr'SDE, $\Delta t={step_size}$, $T={max_steps*step_size}$')
fig.show()    
plots.visualize_trajectories(sde_logger.get_trajectory('particles'), max_time=max_steps*step_size)
plots.plot_kl_divergence(sde_logger.get_trajectory('particles'), target_density_obj.density)
# plots.plot_fisher_divergence(sde_logger.get_trajectory('particles'), target_score)

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

plots.visualize_trajectories(svgd_logger.get_trajectory('particles'))
plots.plot_kl_divergence(svgd_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# train initial score model
mlp = models.MLP(d=1)
score_model = models.ResNet(mlp)
optimizer = nnx.Optimizer(score_model, optax.adamw(0.001, 0.9))
for i in range(100):
    if i % 10 == 0:
        print(losses.explicit_score_matching_loss(score_model, prior_sample, prior_score(prior_sample)))
    loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score(prior_sample))
    optimizer.update(grads)
print(losses.explicit_score_matching_loss(score_model, prior_sample, prior_score(prior_sample)))
#%%
# sample with sbtm
sbtm_logger = sampler.Logger()
loss = losses.implicit_score_matching_loss
sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer, gd_stopping_criterion=sampler.AbsoluteLossChange(0.01), mini_batch_size=num_particles, heun=False)
sbtm_sample = sbtm_sampler.sample()

mean_num_gd_steps = jnp.mean(jnp.array([len(loss_values) for loss_values in sbtm_logger.get_trajectory('batch_loss_values')]))
print(f'Mean number of GD steps: {mean_num_gd_steps}')
#%%
fig, ax = plots.plot_distributions(prior_sample, sbtm_sample, target_density_obj)
ax.set_title(fr'SBTM, $\Delta t={step_size}$, $T={max_steps*step_size}$')
ax.set_xlim(-10, 10)
fig.show()

sbtm_particles = sbtm_logger.get_trajectory('particles')
sbtm_score = sbtm_logger.get_trajectory('score')
plots.visualize_trajectories(sbtm_particles, title=fr"SBTM, $\Delta t=0.1$ if $t < 10$, $\Delta t=1$ if $t \geq 10$, $T={max_steps*step_size}$")
plots.plot_kl_divergence(sbtm_particles, target_density_obj.density)
plots.plot_fisher_divergence(sbtm_particles, target_score, sbtm_score, yscale='log')

#%%
# Compute and plot the Frobenius norm of the difference of covariance between SDE and analytical solution f
sde_cov_diff = []
sbtm_cov_diff = []

for t_idx, t in enumerate(jnp.linspace(t0, t0 + max_steps * step_size, max_steps)):
    K_t = K(t)
    analytic_cov = jnp.array([[K_t]])
    
    sde_particles_t = sde_logger.get_trajectory('particles')[t_idx]
    sde_cov_t = jnp.cov(sde_particles_t, rowvar=False)
    sde_cov_diff.append(jnp.linalg.norm(sde_cov_t - analytic_cov, ord='fro'))
    
    sbtm_particles_t = sbtm_logger.get_trajectory('particles')[t_idx]
    sbtm_cov_t = jnp.cov(sbtm_particles_t, rowvar=False)
    sbtm_cov_diff.append(jnp.linalg.norm(sbtm_cov_t - analytic_cov, ord='fro'))

fig, ax = plt.subplots(figsize=(6, 6))
plots.plot_quantity_over_time(ax, sde_cov_diff, label='SDE', yscale='log', max_time=max_steps*step_size)
plots.plot_quantity_over_time(ax, sbtm_cov_diff, label='SBTM', yscale='log', max_time=max_steps*step_size)
ax.set_ylabel(rf'$||\Sigma_t - \Sigma_{{\text{{analytic}}}}||_F$')
ax.set_title('Covariance comparision with analytic solution')

# %%
# Compute and plot the explicit score matching loss of sbtm using the score of the analytic solution
analytic_score = lambda t, x: -x / K(t)
sbtm_score = sbtm_logger.get_trajectory('score')

sbtm_explicit_loss = []
for t_idx, t in enumerate(jnp.linspace(t0, t0 + max_steps * step_size, max_steps)):
    sbtm_particles_t = sbtm_logger.get_trajectory('particles')[t_idx]
    analytic_score_t = analytic_score(t, sbtm_particles_t)
    loss_value = jnp.square(sbtm_score[t_idx] - analytic_score_t).mean()
    sbtm_explicit_loss.append(loss_value)

fig, ax = plt.subplots(figsize=(6, 6))
plots.plot_quantity_over_time(ax, sbtm_explicit_loss, label='', yscale='linear', max_time=max_steps*step_size)
ax.set_ylabel(r'$\frac{1}{n} \sum_i ||\nabla \log f_t^*(x_i) - s_t(x_i)||^2$')
ax.set_title('Explicit Score Matching Loss of SBTM over time')
fig.show()