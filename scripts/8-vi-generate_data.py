#%%
"""Generate the data for several examples"""

import importlib
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler, stats, distribution
from flax import nnx
import optax
from tqdm import tqdm
import pickle
import os

# reload modules
for module in [density, plots, kernel, losses, models, sampler, stats, distribution]:
    importlib.reload(module)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.9'


# TODO: for dilation annealing, I need to initialize particles closer to a dirac delta. Otherwise, they do not feel the gradient of the annealed target distribution.

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

def run_sbtm(prior_sample, target_score, step_size, max_steps, prior_score=None):
    # train initial score model
    score_model = models.ResNet(models.MLP(d=prior_sample.shape[1]))
    optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
    prior_score_values = prior_score(prior_sample)
    for i in tqdm(range(10001), desc='Training initial NN', leave=False):
        loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
        optimizer.update(grads)
        if loss_value < 1e-4:
            break
    print(f"\nTrained initial NN in {i} iterations, loss: {loss_value}\n")

    # sample
    logger = sampler.Logger()
    sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, logger, score_model, losses.implicit_score_matching_loss, optimizer, gd_stopping_criterion=sampler.FixedNumBatches(10)).sample()
    return logger

def run_sde(prior_sample, target_score, step_size, max_steps, **kwargs):
    logger = sampler.Logger()
    sampler.SDESampler(prior_sample, target_score, step_size, max_steps, logger).sample()
    return logger

def save_logs(logger, example_name, method_name, annealing_name, step_size, max_steps):
    log_data = {
        'logs': logger.logs,
        'hyperparameters': logger.hyperparameters
    }
    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl'), 'wb') as f:
        pickle.dump(log_data, f)

def train_and_save_model(d):
    num_particles = 10000
    key = jrandom.key(47)

    # prior
    prior_sample = jrandom.multivariate_normal(key, jnp.zeros(d), jnp.eye(d), shape=(num_particles, ))
    prior_score = lambda x: -x

    # train
    score_model = models.ResNet(models.MLP(d=d, hidden_units=[128, 128]))
    optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))
    prior_score_values = prior_score(prior_sample)
    for i in range(10001):
        loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_sample, prior_score_values)
        optimizer.update(grads)
        if loss_value < 1e-4:
            break
        if i % 100 == 0:
            print(f"i: {i}, loss: {loss_value}")

    # save
    path = os.path.expanduser(f'~/SBTM-sampling/data/models/standard_gaussian/d_{d}/hidden_128_128')
    models.save_model(score_model, path)

# #%%
# """Mixture of gaussians"""
# # Initial sample
# importlib.reload(sampler)
# num_particles = 10000
# key = jrandom.key(47)

# prior_dist = distribution.Gaussian(jnp.array([0]), jnp.array([[1]]))
# prior_sample = prior_dist.sample(key, size=num_particles)
# prior_density = prior_dist.density
# prior_score = prior_dist.score

# # target setup
# target_dist_far = distribution.GaussianMixture(means=[-4, 4], covariances=[1, 1], weights=[0.25, 0.75])
# target_dist_near = distribution.GaussianMixture(means=[-2, 2], covariances=[1, 1], weights=[0.25, 0.75])

# # sample
# for (target_dist, example_name) in tqdm([(target_dist_far, 'gaussians_far'), (target_dist_near, 'gaussians_near')], desc='Mixture of gaussians'):
#     # print(f"{example_name}")
#     target_score = target_dist.score

#     # for (step_size, max_steps) in tqdm([(0.01, 1000), (0.01, 10000), (0.01, 100000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000), (0.1, 100000)], desc=f"{example_name}", leave=False):
#     for (step_size, max_steps) in tqdm([(0.01, 1000), (0.01, 10000), (0.1, 100), (0.1, 1000)], desc=f"{example_name}", leave=False):
#         # print(f"    Step size={step_size}, Max steps={max_steps}, t_end={step_size * max_steps}")
#         t_end = step_size * max_steps
        
#         geometric_annealed_score = lambda t,x : geometric_mean_score(λ(t, t_end), x, prior_score, target_score)
#         dilation_annealed_score = lambda t,x : dilation_score(λ(t, t_end), x, target_score)
#         non_annealed_score = lambda t,x : target_score(x)
        
#         for (annealed_score, annealing_name) in tqdm([(geometric_annealed_score, 'geometric'), (dilation_annealed_score, 'dilation'), (non_annealed_score, 'non-annealed')], desc=f"{step_size} * {max_steps}", leave=False):
#             for (run_func, method_name) in tqdm([(run_sbtm, 'sbtm'), (run_sde, 'sde')], desc=f"{annealing_name}", leave=False):
#                 try:
#                     print(f"{method_name}")
#                     logger = run_func(prior_sample, annealed_score, step_size, max_steps)
#                     log_data = {
#                         'logs': logger.logs,
#                         'hyperparameters': logger.hyperparameters
#                     }
#                     data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
#                     os.makedirs(data_dir, exist_ok=True)
#                     with open(os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}_particles_{num_particles}.pkl'), 'wb') as f:
#                         pickle.dump(log_data, f)
#                 except Exception as e:
#                     print(e)

#%%
"""Analytic solution"""
importlib.reload(sampler)

example_name = 'analytic'

# initial set up
def K(t):
    return 1 - jnp.exp(-2*t)

t0 = 0.1
num_particles = 10000
key = jrandom.key(47)


# sample
# for (step_size, max_steps) in tqdm([(0.1, 50), (0.05, 100), (0.02, 250), (0.01, 500), (0.005, 1000), (0.002, 2500)], desc='Analytic'):
for (step_size, max_steps) in tqdm([(0.01, 500), (0.002, 2500)], desc='Analytic'):
    # for d in tqdm([1], desc=f'{step_size} * {max_steps}', leave=False):    
    for d in tqdm([3, 5, 10, 20], desc=f'{step_size} * {max_steps}', leave=False):    
        prior_dist = distribution.Gaussian(jnp.zeros(d), jnp.eye(d) * K(t0))
        prior_sample = prior_dist.sample(key, size=num_particles)
        prior_score = prior_dist.score

        target_dist = distribution.Gaussian(jnp.zeros(d), jnp.eye(d))
        target_score = target_dist.score
        for (run_func, method_name) in tqdm([(run_sbtm, 'sbtm'), (run_sde, 'sde')], desc=f'd={d}', leave=False):
            try:
                logger = run_func(prior_sample, target_score, step_size, max_steps, prior_score=prior_score)
                log_data = {
                    'logs': logger.logs,
                    'hyperparameters': logger.hyperparameters
                }
                data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/d_{d}/{method_name}/non-annealed')
                os.makedirs(data_dir, exist_ok=True)
                with open(os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}_particles_{num_particles}.pkl'), 'wb') as f:
                    pickle.dump(log_data, f)
            except Exception as e:
                print(e)

# #%%
# """Mixture of 2-d gaussians"""
# # Initial sample
# num_particles = 10000
# key = jrandom.key(47)

# d = 2
# prior_sample = jrandom.multivariate_normal(key, jnp.zeros(d), jnp.eye(d), shape=(num_particles, ))
# prior_density = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, jnp.zeros(d), jnp.eye(d))
# prior_score = lambda x: -x

# # target setup
# target_density_far = jax.jit(lambda x: sum(1/16 * jax.scipy.stats.multivariate_normal.pdf(x, jnp.array([a, b]), 1) for a in [-3*5, -5, 5, 3*5] for b in [-3*5, -5, 5, 3*5]))

# target_density_near = jax.jit(lambda x: sum(1/16 * jax.scipy.stats.multivariate_normal.pdf(x, jnp.array([a, b]), 1) for a in [-3*2, -2, 2, 3*2] for b in [-3*2, -2, 2, 3*2]))

# # sample
# for (target_density, example_name) in [(target_density_far, 'gaussians_far_2d'), (target_density_near, 'gaussians_near_2d')]:
#     print(f"{example_name}")
#     target_score = lambda x: density.score(target_density, x)

#     for (step_size, max_steps) in [(0.01, 10), (0.01, 100), (0.01, 1000), (0.01, 10000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000)]:
#         print(f"    Step size={step_size}, Max steps={max_steps}, t_end={step_size * max_steps}")
#         t_end = step_size * max_steps
        
#         geometric_annealed_score = lambda t,x : geometric_mean_score(λ(t, t_end), x, prior_score, target_score)
#         dilation_annealed_score = lambda t,x : dilation_score(λ(t, t_end), x, target_score)
#         non_annealed_score = lambda t,x : target_score(x)
        
#         for (annealed_score, annealing_name) in [(geometric_annealed_score, 'geometric'), (dilation_annealed_score, 'dilation'), (non_annealed_score, 'non-annealed')]:
#             print(f"        {annealing_name}")

#             method_name = 'sbtm'
#             try:
#                 print(f"            {method_name}")
#                 score_model = models.ResNet(models.MLP(d=d, hidden_units=[128, 128, 128, 128]))
#                 score_model = models.load_model(score_model, os.path.expanduser(f'~/SBTM-sampling/data/models/standard_gaussian/d_2/hidden_128_128_128_128'))
#                 logger = sampler.Logger()
#                 sampler.SBTMSampler(prior_sample, annealed_score, step_size, max_steps, logger, score_model, losses.implicit_score_matching_loss, optimizer=nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))).sample()
#                 save_logs(logger, example_name, method_name, annealing_name, step_size, max_steps)
#             except Exception as e:
#                 print(e)

#             method_name = 'sde'
#             try:
#                 print(f"            {method_name}")
#                 logger = sampler.Logger()
#                 sampler.SDESampler(prior_sample, annealed_score, step_size, max_steps, logger).sample()
#                 save_logs(logger, example_name, method_name, annealing_name, step_size, max_steps)
#             except Exception as e:
#                 print(e)

#%%
# """circle distribution"""
# num_particles = 10000
# key = jrandom.key(47)
# d = 2

# # prior
# prior_sample = jrandom.multivariate_normal(key, jnp.zeros(d), jnp.eye(d), shape=(num_particles, ))
# prior_density = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, jnp.zeros(d), jnp.eye(d))
# prior_score = lambda x: -x

# # target
# def target_density(x):
#     assert x.shape[0] == d
#     return jax.scipy.stats.multivariate_normal.pdf(jnp.linalg.norm(x-jnp.array([4., 0.])) - 1., 0., 0.2)
# def target_log_density(x):
#     assert x.shape[0] == d
#     return jax.scipy.stats.multivariate_normal.logpdf(jnp.linalg.norm(x-jnp.array([4., 0.])) - 1., 0., 0.2)
# target_score = jax.jit(lambda x: density.score_log_density(target_log_density, x))

# # sample
# example_name = 'circle'
# print(f"{example_name}")
# for (step_size, max_steps) in tqdm([(0.01, 10), (0.01, 100), (0.01, 1000), (0.1, 10), (0.1, 100), (0.1, 1000)]):
#     t_end = step_size * max_steps
    
#     method_name = 'sbtm'
#     try:
#         print(f"Step size={step_size}, Max steps={max_steps}, {method_name}")
#         score_model = models.ResNet(models.MLP(d=d, hidden_units=[128, 128, 128, 128]))
#         score_model = models.load_model(score_model, os.path.expanduser(f'~/SBTM-sampling/data/models/standard_gaussian/d_2/hidden_128_128_128_128'))
#         logger = sampler.Logger()
#         sampler.SBTMSampler(prior_sample, target_score, step_size, max_steps, logger, score_model, losses.implicit_score_matching_loss, optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))).sample()
#         save_logs(logger, example_name, method_name, 'non-annealed', step_size, max_steps)
#     except Exception as e:
#         print(e)

#     method_name = 'sde'
#     try:
#         print(f"Step size={step_size}, Max steps={max_steps}, {method_name}")
#         logger = sampler.Logger()
#         sampler.SDESampler(prior_sample, target_score, step_size, max_steps, logger).sample()
#         save_logs(logger, example_name, method_name, 'non-annealed', step_size, max_steps)
#     except Exception as e:
#         print(e)
