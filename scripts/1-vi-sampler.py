#%%
import importlib
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler
from flax import nnx
import optax
import os
os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'

#%%
# reload modules
for module in [density, plots, kernel, losses, models, sampler]:
    importlib.reload(module)

#%%
# set up
step_size = 0.1
max_steps = 100
num_particles = 5000
key = jrandom.key(42)

prior_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[10.]])}
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))

target_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[1.]])}
target_density_obj = density.Density(density.gaussian_pdf, target_params)
target_score = target_density_obj.score

#%%
# sample with sde
sde_logger = sampler.Logger()
sde_sampler = sampler.SDESampler(prior_sample, target_score, step_size, max_steps, sde_logger)
sde_sample = sde_sampler.sample()
fig = plots.plot_distributions(prior_sample, sde_sample, target_params)
fig.show()

#%%
# sample with svgd
svgd_kernel = kernel.Kernel(kernel.rbf_kernel)
svgd_logger = sampler.Logger()
svgd_sampler = sampler.SVGDSampler(prior_sample, target_score, step_size, max_steps, svgd_logger, svgd_kernel)
svgd_sample = svgd_sampler.sample()
fig = plots.plot_distributions(prior_sample, svgd_sample, target_params)
fig.show()

#%%
# sample with sbtm
mlp = models.MLP(d=1)
score_model = models.ResNet(mlp)
optimizer = nnx.Optimizer(score_model, optax.adamw(0.01, 0.9))
for i in range(100):
    if i % 10 == 0:
        print(losses.explicit_score_matching_loss(score_model, target_score, prior_sample))
    loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, target_score, prior_sample)
    optimizer.update(grads)
print(losses.explicit_score_matching_loss(score_model, target_score, prior_sample))

#%%
importlib.reload(sampler)
sbtm_logger = sampler.Logger()
loss = losses.implicit_score_matching_loss
sbtm_sampler = sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, sbtm_logger, score_model, loss, optimizer)
sbtm_sample = sbtm_sampler.sample()
fig = plots.plot_distributions(prior_sample, sbtm_sample, target_params)
fig.show()

# %%
import matplotlib.pyplot as plt
loss_values = [loss_value for log in sbtm_logger.logs for loss_value in log['loss_values']]

def exponential_moving_average(data, alpha):
    ema = []
    ema_current = data[0]
    for value in data:
        ema_current = alpha * value + (1 - alpha) * ema_current
        ema.append(ema_current)
    return ema

ema_losses = exponential_moving_average(loss_values, alpha=0.1)

plt.plot(loss_values, label='Losses')
plt.plot(ema_losses, label='Exponential Moving Average', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
