#%%
import os
import pickle 
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
import numpy as np

from sbtm import density, plots, kernel, losses, models, sampler, stats, distribution
for module in [density, plots, kernel, losses, models, sampler, stats, distribution]:
    importlib.reload(module)

#%%
target_distributions = {
    'gaussians_far': distribution.GaussianMixture(means=[-4, 4], covariances=[1, 1], weights=[0.25, 0.75]), 
    'gaussians_near': distribution.GaussianMixture(means=[-2, 2], covariances=[1, 1], weights=[0.25, 0.75]), 
    'analytic': distribution.GaussianMixture(means=[0], covariances=[1], weights=[1]),
    'gaussians_far_2d': distribution.GaussianMixture(
        means=[[-15, -15], [-15, -5], [-15, 5], [-15, 15],
               [-5, -15], [-5, -5], [-5, 5], [-5, 15],
               [5, -15], [5, -5], [5, 5], [5, 15],
               [15, -15], [15, -5], [15, 5], [15, 15]],
        covariances=[1]*16,
        weights=[1/16]*16),
    'gaussians_near_2d': distribution.GaussianMixture(
        means=[[-6, -6], [-6, -2], [-6, 2], [-6, 6],
                [-2, -6], [-2, -2], [-2, 2], [-2, 6],
                [2, -6], [2, -2], [2, 2], [2, 6],
                [6, -6], [6, -2], [6, 2], [6, 6]],
        covariances=[1]*16,
        weights=[1/16]*16)
}

#%%
"1d Gaussian mixtures"

for (step_size, max_steps) in tqdm([(0.1, 100), (0.1, 1000), (0.1, 10000), (0.1, 100000)]):
    for example_name in tqdm(['gaussians_far', 'gaussians_near'], leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
        target_samples = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=1000)
        kde = gaussian_kde(target_samples.T)
        for annealing_name in tqdm(['geometric', 'dilation', 'non-annealed'], leave=False, desc=f'example={example_name}'):
            for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
                try:
                    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
                    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
                    with open(path, 'rb') as f:
                        log_data = pickle.load(f)
                    
                    prior_sample = log_data['logs'][0]['particles']
                    sample = log_data['logs'][-1]['particles']

                    fig, ax = plots.plot_distributions(prior_sample, sample, target_distributions[example_name].density)
                    ax.set_xlim(-10, 10)
                    ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
                    x = jnp.linspace(-10, 10, 1000)
                    ax.plot(x, kde(x), lw=2, label='Target KDE', color='orange')
                    ax.legend()

                    save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
                    plt.savefig(save_path)
                    fig.show()
                    plt.close()
                except:
                    print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')

#%%
"""1d Analytic solution"""

example_name = 'analytic'
annealing_name = 'non-annealed'
for (step_size, max_steps) in tqdm([(0.1, 50), (0.05, 100), (0.02, 250), (0.01, 500), (0.005, 1000), (0.002, 2500)], desc=f'example={example_name}'):
    target_samples = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=1000)
    kde = gaussian_kde(target_samples.T)
    for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
        try:
            data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
            path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
            with open(path, 'rb') as f:
                log_data = pickle.load(f)
            
            prior_sample = log_data['logs'][0]['particles']
            sample = log_data['logs'][-1]['particles']

            fig, ax = plots.plot_distributions(prior_sample, sample, target_distributions[example_name].density)
            ax.set_xlim(-10, 10)
            ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
            x = jnp.linspace(-10, 10, 1000)
            ax.plot(x, kde(x), lw=2, label='Target KDE', color='orange')
            ax.legend()

            save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
            plt.savefig(save_path)
            fig.show()
            plt.close()
        except:
            print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
            
#%%
"2d Gaussian mixtures"
importlib.reload(plots)

# example_name = 'gaussians_far_2d'
# method_name = 'sde'
# annealing_name = 'non-annealed'
# step_size = 0.1
# max_steps = 100
# data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
# path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
# with open(path, 'rb') as f:
#     log_data = pickle.load(f)

# sample = log_data['logs'][-1]['particles']
# fig, ax = plots.plot_distributions_2d(sample, target_distributions[example_name].density, lims=[[-20, 20], [-20, 20]])
# fig.show()

lims_near = [[-8, 8], [-8, 8]]
lims_far = [[-20, 20], [-20, 20]]
for (step_size, max_steps) in tqdm([(0.01, 10), (0.01, 100), (0.01, 1000), (0.01, 10000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000)], desc=f'example={example_name}'):
    for (example_name, lims) in tqdm(zip(['gaussians_far_2d', 'gaussians_near_2d'], [lims_far, lims_near]), leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
        for annealing_name in tqdm(['geometric', 'dilation', 'non-annealed'], leave=False, desc=f'example={example_name}'):
            for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
                try:
                    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
                    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
                    with open(path, 'rb') as f:
                        log_data = pickle.load(f)
                    
                    sample = log_data['logs'][-1]['particles']

                    fig, ax = plots.plot_distributions_2d(sample, target_distributions[example_name].density, lims=lims)
                    ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')

                    save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
                    plt.savefig(save_path)
                    # fig.show()
                    plt.close()
                except:
                    print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
