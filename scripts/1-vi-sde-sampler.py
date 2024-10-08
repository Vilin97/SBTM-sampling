import jax.numpy as jnp
import jax.random as jrandom
from sbtm import sampler, density, plots
from sbtm.sampler import SDESampler, Logger

# set up
key = jrandom.key(42)
num_particles = 5000
prior_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[10.]])}
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))

target_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[1.]])}
target_density_obj = density.Density(density.gaussian_pdf, target_params)
target_score = target_density_obj.score

logger = Logger()

# sample
step_size = 0.1
max_steps = 100
sampler = SDESampler(prior_sample, target_score, step_size, max_steps, logger)
sample = sampler.sample()

# plot
fig = plots.plot_distributions(prior_sample, sample, target_params)
fig.show()
