import jax.numpy as jnp
import jax.random as jrandom
from sbtm import sampler, density, plots, kernel
from sbtm.sampler import SDESampler, Logger

import os
os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'

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

# sample with sde
sde_logger = Logger()
sde_sampler = SDESampler(prior_sample, target_score, step_size, max_steps, sde_logger)
sde_sample = sde_sampler.sample()
fig = plots.plot_distributions(prior_sample, sde_sample, target_params)
fig.show()

# sample with svgd
svgd_kernel = kernel.Kernel(kernel.rbf_kernel)
svgd_logger = Logger()
svgd_sampler = sampler.SVGDSampler(prior_sample, target_score, step_size, max_steps, svgd_logger, svgd_kernel)
svgd_sample = svgd_sampler.sample()
fig = plots.plot_distributions(prior_sample, svgd_sample, target_params)
fig.show()
