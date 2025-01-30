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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.9'
# Set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
target_distributions = {
    'gaussians_far': distribution.GaussianMixture(means=[-4, 4], covariances=[1, 1], weights=[0.25, 0.75]),  # T = 10 is enough for metastability, and T=10,000 for full convergence
    'gaussians_near': distribution.GaussianMixture(means=[-2, 2], covariances=[1, 1], weights=[0.25, 0.75]), # T = 100 is enough
    'analytic': distribution.Gaussian(0, 1),                                                                 # T = 5 is enough
    'gaussians_far_2d': distribution.GaussianMixture(                                                        # T = 10 is enough for non-annealed, but there is no convergence
        means=[[-15, -15], [-15, -5], [-15, 5], [-15, 15],
               [-5, -15], [-5, -5], [-5, 5], [-5, 15],
               [5, -15], [5, -5], [5, 5], [5, 15],
               [15, -15], [15, -5], [15, 5], [15, 15]],
        covariances=[1]*16,
        weights=[1/16]*16),
    'gaussians_near_2d': distribution.GaussianMixture(                                                       # T = 100 is enough for non-annealed
        means=[[-6, -6], [-6, -2], [-6, 2], [-6, 6],
                [-2, -6], [-2, -2], [-2, 2], [-2, 6],
                [2, -6], [2, -2], [2, 2], [2, 6],
                [6, -6], [6, -2], [6, 2], [6, 6]],
        covariances=[1]*16,
        weights=[1/16]*16),
    'circle': distribution.Circle(center=[4, 0], radius=1, noise=0.2)
}

#%%
def plot_entropy_dissipation(example_name, target_dist, step_size, max_steps, smoothing=0.5, save=True, num_particles=1000):
    num_particles_str = f'_particles_10000' if num_particles == 10000 else ''
    annealing_name = 'non-annealed'
    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/sbtm/{annealing_name}')
    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}.pkl')
    with open(path, 'rb') as f:
        log_data = pickle.load(f)
    sbtm_particles = jnp.array([log['particles'] for log in log_data['logs']])
    sbtm_scores = jnp.array([log['score'] for log in log_data['logs']])

    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/sde/{annealing_name}')
    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}.pkl')
    with open(path, 'rb') as f:
        log_data = pickle.load(f)
    sde_particles = jnp.array([log['particles'] for log in log_data['logs']])

    fig, ax = plt.subplots(figsize=(10, 6))
    T = max_steps*step_size

    # relative entropy dissipation
    kl_div_time_derivative_sde = -stats.time_derivative(stats.compute_kl_divergences(sde_particles, target_dist.log_density), step_size)
    kl_div_time_derivative_sde = jnp.where(jnp.isnan(kl_div_time_derivative_sde), jnp.nanmax(kl_div_time_derivative_sde), kl_div_time_derivative_sde)
    kl_div_time_derivative_sde = jnp.clip(kl_div_time_derivative_sde, a_min=1e-5, a_max=1e4)
    plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative_sde, smoothing), label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, SDE', marker='o', markersize=3, max_time=T)

    kl_div_time_derivative_sbtm = -stats.time_derivative(stats.compute_kl_divergences(sbtm_particles, target_dist.log_density), step_size)
    kl_div_time_derivative_sbtm = jnp.where(jnp.isnan(kl_div_time_derivative_sbtm), jnp.nanmax(kl_div_time_derivative_sbtm), kl_div_time_derivative_sbtm)
    kl_div_time_derivative_sbtm = jnp.clip(kl_div_time_derivative_sbtm, a_min=1e-5, a_max=1e4)
    plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative_sbtm, smoothing), label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, SBTM', marker='o', markersize=3, max_time=T)

    # relative fisher info
    sbtm_fisher_divs = jnp.array(stats.compute_fisher_divergences(sbtm_particles, sbtm_scores, target_dist.score))
    plots.plot_quantity_over_time(ax, stats.ema(sbtm_fisher_divs, smoothing), label=r'$\frac{1}{n}\sum_{i=1}^n\|\nabla \log \pi_t(X_i) - s(X_i)\|^2$, SBTM', max_time=T)
    ax.set_yscale('log')
    ax.set_title(f"{example_name} $\Delta t={step_size}$, $T={T}$")

    if save:
        save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/entropy_dissipation')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}{num_particles_str}.png'))
    return fig, ax

#%%
# "entropy dissipation plots"

# for (step_size, max_steps) in tqdm([(0.01, 1000), (0.01, 10000), (0.01, 100000)], desc='1d Gaussian mixtures'):
#     for example_name in tqdm(['gaussians_far', 'gaussians_near'], leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
#         try:
#             plot_entropy_dissipation(example_name, target_distributions[example_name], step_size, max_steps, num_particles=10000)
#         except Exception as e:
#             print(f'\n Entropy dissipation plot failed for {example_name}, {step_size}, {max_steps}: \n{e}')
        
# plt.close('all') 
# #%%
# "1d Gaussian mixtures"

# lims_near = [-8, 8]
# lims_far = [-10, 10]
# # for (step_size, max_steps) in tqdm([(0.01, 1000), (0.01, 10000), (0.01, 100000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000), (0.1, 100000)], desc='1d Gaussian mixtures'):
# for (step_size, max_steps) in tqdm([(0.01, 1000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 100000)], desc='1d Gaussian mixtures'):
#     for (example_name, lims) in tqdm(list(zip(['gaussians_far', 'gaussians_near'], [lims_far, lims_near])), leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
#         target_samples = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=1000)
#         kde = gaussian_kde(target_samples.T)
#         target_dist = target_distributions[example_name]
#         for annealing_name in tqdm(['geometric', 'dilation', 'non-annealed'], leave=False, desc=f'example={example_name}'):
#             for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
#                 try:
#                     data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
#                     path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
#                     with open(path, 'rb') as f:
#                         log_data = pickle.load(f)
                        
#                     all_particles = jnp.array([log['particles'] for log in log_data['logs']])

#                     # initial and final particles
#                     prior_sample = all_particles[0]
#                     sample = all_particles[-1]
                    
#                     # fig, ax = plots.plot_distributions(prior_sample, sample, target_dist.density, lims=lims)
#                     # ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
#                     # x = jnp.linspace(lims[0], lims[1], 1000)
#                     # ax.plot(x, kde(x), lw=2, label='Target KDE', color='orange')
#                     # ax.legend()

#                     save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
#                     # os.makedirs(save_dir, exist_ok=True)
#                     # save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
#                     # plt.savefig(save_path)
                    
#                     # trajectories
#                     fig, ax = plots.visualize_trajectories(all_particles, particle_idxs = [0,1], max_time=step_size*max_steps, title=f'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
#                     save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}_trajectories.png')
#                     plt.savefig(save_path)
                    
#                     plt.close('all')
#                 except:
#                     print(f'\nFailed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
        
# #%%
# """1d Analytic solution"""

# example_name = 'analytic'
# annealing_name = 'non-annealed'
# lims = [-5, 5]
# for (step_size, max_steps) in tqdm([(0.1, 50), (0.05, 100), (0.02, 250), (0.01, 500), (0.005, 1000), (0.002, 2500)], desc=f'{example_name}'):
#     target_samples = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=1000)
#     kde = gaussian_kde(target_samples.T)
#     for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
#         try:
#             data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
#             path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
#             with open(path, 'rb') as f:
#                 log_data = pickle.load(f)
                
#             all_particles = jnp.array([log['particles'] for log in log_data['logs']])

#             # initial and final particles
#             prior_sample = all_particles[0]
#             sample = all_particles[-1]

#             # fig, ax = plots.plot_distributions(prior_sample, sample, target_distributions[example_name].density, lims=lims)
#             # ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
#             # x = jnp.linspace(lims[0], lims[1], 1000)
#             # ax.plot(x, kde(x), lw=2, label='Target KDE', color='orange')
#             # ax.legend()

#             save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
#             # os.makedirs(save_dir, exist_ok=True)
#             # save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
#             # plt.savefig(save_path)
            
#             # trajectories
#             fig, ax = plots.visualize_trajectories(all_particles, particle_idxs = [0,1], max_time=step_size*max_steps, title=f'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
#             save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}_trajectories.png')
#             plt.savefig(save_path)
            
#             plt.close('all')
#         except:
#             print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
            
# # #%%
# # "2d Gaussian mixtures"

# # lims_near = [[-8, 8], [-8, 8]]
# # lims_far = [[-20, 20], [-20, 20]]
# # for (step_size, max_steps) in tqdm([(0.01, 10), (0.01, 100), (0.01, 1000), (0.01, 10000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000)], desc=f'2d Gaussian mixtures'):
# #     for (example_name, lims) in tqdm(list(zip(['gaussians_far_2d', 'gaussians_near_2d'], [lims_far, lims_near])), leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
# #         for annealing_name in tqdm(['geometric', 'dilation', 'non-annealed'], leave=False, desc=f'example={example_name}'):
# #             for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
# #                 try:
# #                     data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
# #                     path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
# #                     with open(path, 'rb') as f:
# #                         log_data = pickle.load(f)
                    
# #                     sample = log_data['logs'][-1]['particles']

# #                     fig, ax = plots.plot_distributions_2d(sample, target_distributions[example_name].density, lims=lims)
# #                     ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')

# #                     save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
# #                     os.makedirs(save_dir, exist_ok=True)
# #                     save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
# #                     plt.savefig(save_path)
# #                     plt.close()
# #                 except:
# #                     print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')

#%%
# "Circle distribution"
# importlib.reload(distribution)

# example_name = 'circle'
# annealing_name = 'non-annealed'
# lims = [[-2, 7], [-4, 4]]
# for (step_size, max_steps) in tqdm([(0.01, 10), (0.01, 100), (0.01, 1000), (0.1, 10), (0.1, 100), (0.1, 1000)], desc=f'Circle distribution'):
#     for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
#         try:
#             data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
#             path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
#             with open(path, 'rb') as f:
#                 log_data = pickle.load(f)
            
#             sample = log_data['logs'][-1]['particles']

#             fig, ax = plots.plot_distributions_2d(sample, target_distributions[example_name].density, lims=lims)
#             ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')

#             save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
#             os.makedirs(save_dir, exist_ok=True)
#             save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
#             plt.savefig(save_path)
#             plt.close()
#         except:
#             print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
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

example_name = 'gaussians_far_2d'
d = 2
annealing_name = 'dilation'

step_size = 0.01
max_steps = 10000

prior_dist = distribution.Gaussian(jnp.zeros(d), jnp.eye(d))

target_dist = target_distributions[example_name]
if annealing_name == 'geometric':
    annealed_score = lambda t,x : geometric_mean_score(λ(t, step_size*max_steps), x, prior_dist.score, target_dist.score)
elif annealing_name == 'dilation':
    annealed_score = lambda t,x : dilation_score(λ(t, step_size*max_steps), x, target_dist.score)
elif annealing_name == 'non-annealed':
    annealed_score = lambda t,x: target_dist.score(x)

data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/sbtm/{annealing_name}')
path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
with open(path, 'rb') as f:
    log_data = pickle.load(f)
particles = jnp.array([log['particles'] for log in log_data['logs']])
scores = jnp.array([log['score'] for log in log_data['logs']])

# compute d/dt KL
kl_div_time_derivative = -stats.time_derivative(stats.compute_kl_divergences(particles, target_dist.log_density), step_size)
kl_div_time_derivative = jnp.where(jnp.isnan(kl_div_time_derivative), jnp.nanmax(kl_div_time_derivative), kl_div_time_derivative)
kl_div_time_derivative = jnp.clip(kl_div_time_derivative, a_min=1e-5, a_max=1e4)

# compute relative Fisher info
fisher_divs = []
ts = jnp.linspace(0.0, step_size*max_steps, max_steps)
for ti, particles_i, scores_i in list(zip(ts, particles, scores)):
    value = jnp.mean(jax.vmap(lambda x,y: jnp.dot(x, y))(scores_i - target_dist.score(particles_i), scores_i - annealed_score(ti, particles_i)))
    fisher_divs.append(value)

#%%
save = False
for (steps_to_plot, smoothing) in zip([1000, 2000, 5000, 10000], [0.8, 0.9, 0.95, 0.99]):

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))    
    T = steps_to_plot*step_size
    plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, SBTM', markersize=3, max_time=T)
    plots.plot_quantity_over_time(ax, stats.ema(fisher_divs, smoothing)[:steps_to_plot], label=fr'$\frac{{1}}{{n}}\sum_{{i=1}}^n \langle s(X_i) - \nabla \log \pi(X_i),  s(X_i) - \nabla \log \pi_t(X_i) \rangle$, SBTM', max_time=T)
    ax.set_title(f"{example_name} Δt = {step_size}, T={T} {annealing_name}")

    fig.show()
    if save:
        save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/entropy_dissipation/{annealing_name}')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{steps_to_plot}_particles_10000.png'))