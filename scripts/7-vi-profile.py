import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

import jax
import importlib
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler, stats
from flax import nnx
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
import pickle
# reload modules
for module in [density, plots, kernel, losses, models, sampler, stats]:
    importlib.reload(module)

# TODO: profile the sampler to see what can be sped up

step_size = 0.01
max_steps = 100
t_end = step_size * max_steps
num_particles = 1000
key = jrandom.key(42)

prior_params = {'mean': jnp.array([0]), 'covariance': jnp.array([[1.]])}
prior_density_obj = density.Density(density.gaussian_pdf, prior_params)
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))
prior_score = density.Density(density.gaussian_pdf, prior_params).score


def target_density(x):
    return 0.1 * jax.scipy.stats.norm.pdf(x, -5, 1) + 0.9 * jax.scipy.stats.norm.pdf(x, 5, 1)

target_score = lambda x: density.score(lambda y: target_density(y)[0], x)


t_end = step_size * max_steps
def λ(t):
    """interpolate between 0 and 1"""
    t = (t/t_end)**2
    return min((t/0.9), 1)

def annealed_score1(t, x, threshold=0.2):
    # t = t/t_end
    # t = t/(1-threshold)
    t = np.clip(t, threshold, 1)
    return target_score(x/t)

annealed_score = lambda t,x : annealed_score1(λ(t), x)

sde_logger = sampler.Logger()
sde_sampler = sampler.SDESampler(prior_sample, annealed_score, step_size, max_steps, sde_logger)
sde_sample = sde_sampler.sample()