#%%
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler
from flax import nnx
import optax
import os
import time
import timeit
import optimistix as optx

os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'
import matplotlib.pyplot as plt

for module in [density, plots, kernel, losses, models, sampler]:
    importlib.reload(module)

#%%
# set up
step_size = [0.01]*100 + [0.1]*100 + [1]*100 #+ [10]*1000
max_steps = len(step_size)
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
plots.visualize_trajectories(sde_logger.get_trajectory('particles'), particle_idxs=[])
plots.plot_kl_divergence(sde_logger.get_trajectory('particles'), target_density_obj.density)

#%%
# train initial score model
mlp = models.MLP(d=1)
score_model = models.ResNet(mlp)
optimizer = nnx.Optimizer(score_model, optax.adamw(0.0004))
for i in range(101):
    if i % 50 == 0:
        print(losses.explicit_score_matching_loss(score_model, prior_sample, prior_score(prior_sample)))
    loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score(prior_sample))
    optimizer.update(grads)

# plot the trained model and the true score
x = jnp.linspace(-10, 10, 1000).reshape(-1, 1)

plt.plot(x, prior_score(x), label='True Score', linestyle='--')
plt.plot(x, score_model(x), label='Trained Score', linestyle='-')
plt.legend()
plt.show()

# %%
# sample with sbtm
sbtm_logger = sampler.Logger()
loss = losses.implicit_score_matching_loss
sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, 0.1, 100, sbtm_logger, score_model, loss, optimizer)
sbtm_sample = sbtm_sampler.sample()
#%%
fig, ax = plots.plot_distributions(prior_sample, sbtm_sample, target_density_obj)
ax.set_title('SBTM')
ax.set_xlim(-10, 10)
fig.show()
plots.visualize_trajectories(sbtm_logger.get_trajectory('particles'))
plots.plot_kl_divergence(sbtm_logger.get_trajectory('particles'), target_density_obj.density)
plots.plot_fisher_divergence(sbtm_logger.get_trajectory('particles'), target_score, sbtm_logger.get_trajectory('score'))
loss_values = [loss_value for log in sbtm_logger.logs for loss_value in log['loss_values']]
batch_loss_values = [loss_value for log in sbtm_logger.logs for loss_value in log['batch_loss_values']]
plots.plot_losses(loss_values, batch_loss_values)

#%%
def fn(x, args):
    return prior_sample + 0.9 * (target_score(x) + prior_score(x))
rtol=1e-2
atol=1e-2
solver=optx.FixedPointIteration(rtol=rtol, atol=atol)
sol=optx.fixed_point(fn, solver, prior_sample)
sol.stats['num_steps']
        
# solvers = {
#     'FixedPointIteration': optx.FixedPointIteration(rtol=rtol, atol=atol)
# }

# for name, solver in solvers.items():
#     try:
#         start_time = time.time()
#         sol = optx.fixed_point(fn, solver, prior_sample)
#         end_time = time.time()
#         duration = end_time - start_time
#         print(f"{name} solver took {duration:.4f} seconds with {sol.stats['num_steps']} steps")
#     except Exception as e:
#         print(f"ERROR!")
# %%