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

# Set the memory fraction for JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.45'
# Set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        weights=[1/16]*16),
    'circle': distribution.Circle(center=[4, 0], radius=1, noise=0.2)
}

#%%
def plot_entropy_dissipation(example_name, target_dist, step_size, max_steps, smoothing=0.1):
    annealing_name = 'non-annealed'
    try:
        data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/sbtm/{annealing_name}')
        path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
        with open(path, 'rb') as f:
            log_data = pickle.load(f)
        sbtm_particles = jnp.array([log['particles'] for log in log_data['logs']])
        sbtm_scores = jnp.array([log['score'] for log in log_data['logs']])

        data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/sde/{annealing_name}')
        path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
        with open(path, 'rb') as f:
            log_data = pickle.load(f)
        sde_particles = jnp.array([log['particles'] for log in log_data['logs']])

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_every_n = max(1, max_steps // 500)
        T = plot_every_n*step_size

        # relative entropy dissipation
        for particles, method_name in zip([sde_particles, sbtm_particles], ['SDE', 'SBTM']):
            kl_div_time_derivative = -stats.time_derivative(stats.compute_kl_divergences(particles, target_dist.density), step_size)
            plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative[::plot_every_n], smoothing), label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, {method_name}', marker='o', markersize=3, max_time=T)

        # relative fisher info
        sbtm_fisher_divs = jnp.array(stats.compute_fisher_divergences(sbtm_particles, sbtm_scores, target_dist.score))
        plots.plot_quantity_over_time(ax, stats.ema(sbtm_fisher_divs[::plot_every_n], smoothing), label=r'$\frac{1}{n}\sum_{i=1}^n\|\nabla \log \pi_t(X_i) - s(X_i)\|^2$, SBTM', max_time=T)
        ax.set_yscale('log')
        ax.set_title(f"{example_name} $\Delta t={step_size}$, $T={T}$")
        
        # save
        save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/entropy_dissipation')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png'))
        plt.close()
    except:
        print(f'\n Entropy dissipation plot failed for {example_name}, {annealing_name}, {step_size}, {max_steps}')

#%%
"1d Gaussian mixtures"

lims_near = [-8, 8]
lims_far = [-10, 10]
for (step_size, max_steps) in tqdm([(0.01, 1000), (0.01, 10000), (0.01, 100000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000), (0.1, 100000)], desc='1d Gaussian mixtures'):
    for (example_name, lims) in tqdm(list(zip(['gaussians_far', 'gaussians_near'], [lims_far, lims_near])), leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
        target_samples = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=1000)
        kde = gaussian_kde(target_samples.T)
        target_dist = target_distributions[example_name]
        for annealing_name in tqdm(['geometric', 'dilation', 'non-annealed'], leave=False, desc=f'example={example_name}'):
            if annealing_name == 'non-annealed':
                plot_entropy_dissipation(example_name, target_dist, step_size, max_steps)
            
            for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
                try:
                    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
                    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
                    with open(path, 'rb') as f:
                        log_data = pickle.load(f)
                        
                    all_particles = jnp.array([log['particles'] for log in log_data['logs']])

                    # initial and final particles
                    prior_sample = all_particles[0]
                    sample = all_particles[-1]
                    
                    fig, ax = plots.plot_distributions(prior_sample, sample, target_dist.density, lims=lims)
                    ax.set_xlim(-10, 10)
                    ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
                    x = jnp.linspace(-10, 10, 1000)
                    ax.plot(x, kde(x), lw=2, label='Target KDE', color='orange')
                    ax.legend()

                    save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
                    plt.savefig(save_path)
                    
                    # trajectories
                    fig, ax = plots.visualize_trajectories(all_particles, particle_idxs = [0,1], max_time=step_size*max_steps, title=f'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
                    save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}_trajectories.png')
                    plt.savefig(save_path)
                    
                    plt.close('all')
                except:
                    print(f'\nFailed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')

#%%
"""1d Analytic solution"""

example_name = 'analytic'
annealing_name = 'non-annealed'
for (step_size, max_steps) in tqdm([(0.1, 50), (0.05, 100), (0.02, 250), (0.01, 500), (0.005, 1000), (0.002, 2500)], desc=f'example={example_name}'):
    target_samples = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=1000)
    kde = gaussian_kde(target_samples.T)
    if annealing_name == 'non-annealed':
        plot_entropy_dissipation(example_name, target_dist, step_size, max_steps)
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
            plt.close()
        except:
            print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
            
#%%
"2d Gaussian mixtures"
importlib.reload(plots)

lims_near = [[-8, 8], [-8, 8]]
lims_far = [[-20, 20], [-20, 20]]
for (step_size, max_steps) in tqdm([(0.01, 10), (0.01, 100), (0.01, 1000), (0.01, 10000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000)], desc=f'2d Gaussian mixtures'):
    for (example_name, lims) in tqdm(list(zip(['gaussians_far_2d', 'gaussians_near_2d'], [lims_far, lims_near])), leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
        for annealing_name in tqdm(['geometric', 'dilation', 'non-annealed'], leave=False, desc=f'example={example_name}'):
            if annealing_name == 'non-annealed':
                plot_entropy_dissipation(example_name, target_dist, step_size, max_steps)
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
                    plt.close()
                except:
                    print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')

#%%
"Circle distribution"
importlib.reload(distribution)

example_name = 'circle'
annealing_name = 'non-annealed'

for (step_size, max_steps) in tqdm([(0.01, 10), (0.01, 100), (0.01, 1000), (0.01, 10000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000)], desc=f'Circle distribution'):
    if annealing_name == 'non-annealed':
        plot_entropy_dissipation(example_name, target_dist, step_size, max_steps)
    for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
        try:
            data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
            path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
            with open(path, 'rb') as f:
                log_data = pickle.load(f)
            
            sample = log_data['logs'][-1]['particles']

            fig, ax = plots.plot_distributions_2d(sample, target_distributions[example_name].density)
            ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')

            save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
            plt.savefig(save_path)
            plt.close()
        except:
            print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
