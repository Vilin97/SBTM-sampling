
#%%
"""Generate the data for several examples"""

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
import os

# reload modules
for module in [density, plots, kernel, losses, models, sampler, stats]:
    importlib.reload(module)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#%%
def λ(t, t_end):
    """interpolate between 0 and 1"""
    t = (t/t_end)
    return t

def dilation_score(t, x, target_score, threshold=0.2):
    t = np.clip(t, threshold, 1)
    return target_score(x/t)

def geometric_mean_score(t, x, prior_score, target_score):
    return t * target_score(x) + (1-t) * prior_score(x)

def run_sbtm(prior_sample, target_score, step_size, max_steps):
    # train initial score model
    score_model = models.ResNet(models.MLP(d=prior_sample.shape[1]))
    optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
    prior_score_values = prior_score(prior_sample)
    # print("Training initial NN. Losses:")
    for i in range(101):
        loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
        optimizer.update(grads)
        # if i % 50 == 0:
        #     print(loss_value)

    # sample
    logger = sampler.Logger()
    sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, logger, score_model, losses.implicit_score_matching_loss, optimizer, gd_stopping_criterion=sampler.FixedNumBatches(10)).sample()
    return logger

def run_sde(prior_sample, target_score, step_size, max_steps):
    logger = sampler.Logger()
    sampler.SDESampler(prior_sample, target_score, step_size, max_steps, logger).sample()
    return logger

#%%
# Initial sample
importlib.reload(sampler)
num_particles = 1000
key = jrandom.key(47)

prior_sample = jrandom.multivariate_normal(key, jnp.array([0]), jnp.array([[1.]]), shape=(num_particles,))
prior_density = lambda x: jax.scipy.stats.norm.pdf(x, 0, 1)
prior_score = lambda x: density.score(lambda y: prior_density(y)[0], x)

# target setup
target_density_far = lambda x: 0.25 * jax.scipy.stats.norm.pdf(x, -4, 1) + 0.75 * jax.scipy.stats.norm.pdf(x, 4, 1)
target_density_near = lambda x: 0.25 * jax.scipy.stats.norm.pdf(x, -4, 1) + 0.75 * jax.scipy.stats.norm.pdf(x, 4, 1)

# sample
for (target_density, example_name) in [(target_density_far, 'gaussians_far'), (target_density_near, 'gaussians_near')]:
    print(f"{example_name}")
    target_score = lambda x: density.score(lambda y: target_density(y)[0], x)

    for (step_size, max_steps) in [(0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000), (0.1, 100000)]:
        print(f"    Step size={step_size}, Max steps={max_steps}, t_end={step_size * max_steps}")
        t_end = step_size * max_steps
        
        geometric_annealed_score = lambda t,x : geometric_mean_score(λ(t, t_end), x, prior_score, target_score)
        dilation_annealed_score = lambda t,x : dilation_score(λ(t, t_end), x, target_score)
        non_annealed_score = lambda t,x : target_score(x)
        
        for (annealed_score, annealing_name) in [(geometric_annealed_score, 'geometric'), (dilation_annealed_score, 'dilation'), (non_annealed_score, 'non-annealed')]:
            for (run_func, method_name) in [(run_sbtm, 'sbtm'), (run_sde, 'sde')]:
                try:
                    print(f"        {method_name} with {annealing_name}")
                    logger = run_func(prior_sample, annealed_score, step_size, max_steps)
                    log_data = {
                        'logs': logger.logs,
                        'hyperparameters': logger.hyperparameters
                    }
                    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
                    os.makedirs(data_dir, exist_ok=True)
                    with open(os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl'), 'wb') as f:
                        pickle.dump(log_data, f)
                except Exception as e:
                    print(e)
