#%%
import importlib
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler, stats
from flax import nnx
import optax
import os
# os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'
import matplotlib.pyplot as plt
import copy

#%%
# reload modules
for module in [density, plots, kernel, losses, models, sampler, stats]:
    importlib.reload(module)

#%%
# set up
step_size = 1.
max_steps = 4000
num_particles = 1000
key = jrandom.key(42)

prior_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[10.]])}
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))
prior_score = density.Density(density.gaussian_pdf, prior_params).score

target_params = {'mean': [jnp.array([-5]), jnp.array([5])], 'covariance': [jnp.array([[1.]]), jnp.array([[1.]])], 'weights': jnp.array([1/10, 1-1/10])}
target_density_obj = density.Density(density.gaussian_mixture_pdf, target_params)
target_score = target_density_obj.score
plt.plot(jnp.linspace(-10, 10, 1000), target_density_obj.density(jnp.reshape(jnp.linspace(-10, 10, 1000), (1000,1))))

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
plots.visualize_trajectories(sde_logger.get_trajectory('particles')[::10], max_time=max_steps*step_size)
plots.plot_kl_divergence(sde_logger.get_trajectory('particles')[::100], target_density_obj.density)
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
optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
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
# sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer, gd_stopping_criterion=sampler.FixedNumBatches(10), mini_batch_size=num_particles)
sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer, gd_stopping_criterion=sampler.AbsoluteLossChange(0.01), mini_batch_size=num_particles)
sbtm_sample = sbtm_sampler.sample()

#%%
fig, ax = plots.plot_distributions(prior_sample, sbtm_sample, target_density_obj)
ax.set_title(fr'SBTM, $\Delta t={step_size}$, $T={max_steps*step_size}$')
ax.set_xlim(-10, 10)
fig.show()

sbtm_particles = sbtm_logger.get_trajectory('particles')
sbtm_score = sbtm_logger.get_trajectory('score')
plots.visualize_trajectories(sbtm_particles, title=fr"SBTM, $\Delta t=0.1$ if $t < 10$, $\Delta t=1$ if $t \geq 10$, $T={max_steps*step_size}$")
plots.plot_kl_divergence(sbtm_particles, target_density_obj.density)
plots.plot_fisher_divergence(sbtm_particles, target_score, sbtm_score)

#%%
from tqdm import tqdm
particles = [jnp.reshape(p, -1) for p in sbtm_logger.get_trajectory('particles')[::1000]]
particle_switches = {}

for i in tqdm(range(len(particles[0]))):
    positions = [particles[t][i] for t in range(len(particles))]
    switches = []
    for t in range(1, len(positions)):
        if positions[t-1] < 0 and positions[t] > 0 or positions[t-1] > 0 and positions[t] < 0:
            switches.append(t)
    if switches:
        particle_switches[i] = switches

print(particle_switches)


#%%
batch_loss_values = sbtm_logger.get_trajectory('batch_loss_values')
num_gd_steps = [len(loss_values) for loss_values in batch_loss_values]
fig, ax = plt.subplots(figsize=(6, 6))
plots.plot_quantity_over_time(ax, num_gd_steps, label='Number of GD steps', plot_zero_line=False, max_time=max_steps*step_size, marker='o', markersize=3)
ax.set_title(f"Number of GD steps, avg={jnp.mean(jnp.array(num_gd_steps)):.1f}")

#%%
loss_values = sbtm_logger.get_trajectory('loss_values')[:10]
pretrain_loss_values = [values[0] for values in loss_values]
post_train_loss_values = [values[-1] for values in loss_values]
zipped = list(zip(pretrain_loss_values, post_train_loss_values))
interleaved = [val for pair in zipped for val in pair]
fig, ax = plt.subplots(figsize=(6, 6))

# plots.plot_quantity_over_time(ax, interleaved, label='interleaved')
plots.plot_quantity_over_time(ax, pretrain_loss_values, label='Pretrain loss')
plots.plot_quantity_over_time(ax, post_train_loss_values, label='Posttrain loss', max_time=10*step_size)

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

#%%
# plot the KL divergence estimates
from scipy import integrate
steps_to_plot = 100
fig, ax = plt.subplots(figsize=(6, 6))
integral_of_fisher_divs_1 = integrate.cumulative_trapezoid(sbtm_fisher_divs, dx=step_size, initial=0.)
integral_of_fisher_divs_2 = jnp.cumulative_sum(jnp.array(sbtm_fisher_divs), include_initial=True) * step_size
sbtm_kl_divs_1 = kde_kl_divs[0] - integral_of_fisher_divs_1
sbtm_kl_divs_2 = kde_kl_divs[0] - integral_of_fisher_divs_2
plots.plot_quantity_over_time(ax, kde_kl_divs[:steps_to_plot], label='KDE estimate of KL', marker='o', markersize=3)
plots.plot_quantity_over_time(ax, sbtm_kl_divs_1[:steps_to_plot], label='NN estimate, trapezoid rule')
plots.plot_quantity_over_time(ax, sbtm_kl_divs_2[:steps_to_plot], label='NN estimate, left Riemann sum', max_time=max_steps*step_size)

ax.set_title("KL Divergence Estimates")
fig.show()

#%%
# Compare the KL divergence of SDE and SBTM samplers over time

#%%
# Annealing: sample from noised target, use the sample as initial condition for less noised target, etc
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
    
# plot trajectories
plots.visualize_trajectories(logger.get_trajectory('particles')[::20], particle_idxs=range(1))

# plot kl divergence
plots.plot_kl_divergence(logger.get_trajectory('particles')[::10], target_density_obj.density)

# plot losses
loss_values = [loss_value for log in logger.logs for loss_value in log['loss_values']]
batch_loss_values = [loss_value for log in logger.logs for loss_value in log['batch_loss_values']]
plots.plot_losses(loss_values, batch_loss_values)

# plot fisher divergence
plots.plot_fisher_divergence(logger.get_trajectory('particles')[::10], target_score, logger.get_trajectory('score')[::10])

# %%
