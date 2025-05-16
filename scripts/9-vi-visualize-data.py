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
# from scipy.stats import gaussian_kde
from jax.scipy.stats import gaussian_kde

from sbtm import density, plots, kernel, losses, models, sampler, stats, distribution
import pickle
from tqdm import tqdm
for module in [density, plots, kernel, losses, models, sampler, stats, distribution]:
    importlib.reload(module)

# Set the memory fraction for JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.9'
# Set the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
def compute_kl_divergence_integral(sample_f, sample_g, num_points=1000, eps=1e-10):
    # Flatten to 1D
    sample_f = sample_f.ravel()
    sample_g = sample_g.ravel()
    
    # Create KDEs
    f_kde = gaussian_kde(sample_f)
    g_kde = gaussian_kde(sample_g)
    
    # Integration grid
    xmin = jnp.minimum(jnp.min(sample_f), jnp.min(sample_g))
    xmax = jnp.maximum(jnp.max(sample_f), jnp.max(sample_g))
    x = jnp.linspace(xmin, xmax, num_points)
    
    # Evaluate KDEs
    f_vals = f_kde.pdf(x)
    g_vals = g_kde.pdf(x)
    
    # KL divergence via trapezoidal rule
    kl = jnp.trapezoid(f_vals * jnp.log(f_vals / (g_vals + eps) + eps), x)
    return kl

def compute_l2_error_integral(sample_f, sample_g, num_points=1000):
    # Flatten to 1D
    sample_f = sample_f.ravel()
    sample_g = sample_g.ravel()
    
    # Create KDEs
    f_kde = gaussian_kde(sample_f)
    g_kde = gaussian_kde(sample_g)
    
    # Integration grid
    xmin = jnp.minimum(jnp.min(sample_f), jnp.min(sample_g))
    xmax = jnp.maximum(jnp.max(sample_f), jnp.max(sample_g))
    x = jnp.linspace(xmin, xmax, num_points)
    
    # Evaluate KDEs
    f_vals = f_kde.pdf(x)
    g_vals = g_kde.pdf(x)
    
    # L2 error via trapezoidal rule
    l2_error = jnp.trapezoid((f_vals - g_vals) ** 2, x)
    return l2_error

def get_particles(example_name, method_name, annealing_name, step_size, max_steps, num_particles):
    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}_particles_{num_particles}.pkl')
    with open(path, 'rb') as f:
        log_data = pickle.load(f)
    all_particles = jnp.array([log['particles'] for log in log_data['logs']])
    return all_particles

#%%
step_size = 0.01
max_steps = 1000

# kls = {}
l2s = {}

for example_name in tqdm(['gaussians_far', 'gaussians_near'], leave=False):
    sample_g = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=10000)
    for method_name in tqdm(['sde', 'sbtm', 'svgd'], leave=False, desc=f'{example_name}'):
        for annealing_name in tqdm(['geometric', 'dilation', 'non-annealed'], leave=False, desc=f'{method_name}'):
            for num_particles in tqdm([100, 300, 1000, 3000, 10000], leave=False, desc=f'{annealing_name}'):
                samples_f = get_particles(example_name, method_name, annealing_name, step_size, max_steps, num_particles)
                print(f"example_name: {example_name}, method_name: {method_name}, annealing_name: {annealing_name}, num_particles: {num_particles}")
                for i in tqdm(range(samples_f.shape[0]), leave=False, desc='time'):
                    # kl = compute_kl_divergence_integral(samples_f[i], sample_g)
                    # key = (example_name, method_name, annealing_name, num_particles, i)
                    # kls[key] = kl.item()
                    # if jnp.isnan(kls[key]):
                    #     print(f"KL is nan for key: {key}")
                        
                    l2 = compute_l2_error_integral(samples_f[i], sample_g)
                    key = (example_name, method_name, annealing_name, num_particles, i)
                    l2s[key] = l2
                    if jnp.isnan(l2s[key]):
                        print(f"L2 is nan for key: {key}")

step_size = 0.002
max_steps = 1250
example_name = 'analytic'
annealing_name = 'non-annealed'
sample_g = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=10000)
for method_name in tqdm(['sde', 'sbtm', 'svgd'], leave=False, desc=f'{example_name}'):
    for num_particles in tqdm([100, 300, 1000, 3000, 10000], leave=False, desc=f'{annealing_name}'):
        data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/d_{1}/{method_name}/non-annealed')
        path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}_particles_{num_particles}.pkl')
        with open(path, 'rb') as f:
            log_data = pickle.load(f)
        samples_f = jnp.array([log['particles'] for log in log_data['logs']])
        print(f"example_name: {example_name}, method_name: {method_name}, annealing_name: {annealing_name}, num_particles: {num_particles}")
        for i in tqdm(range(samples_f.shape[0]), leave=False, desc='time'):
            # kl = compute_kl_divergence_integral(samples_f[i], sample_g)
            # key = (example_name, method_name, annealing_name, num_particles, i)
            # kls[key] = kl
            # if jnp.isnan(kls[key]):
            #     print(f"KL is nan for key: {key}")
            l2 = compute_l2_error_integral(samples_f[i], sample_g)
            key = (example_name, method_name, annealing_name, num_particles, i)
            l2s[key] = l2
            if jnp.isnan(l2s[key]):
                print(f"L2 is nan for key: {key}")

# Save KL divergences to file
# save_path = os.path.expanduser('~/SBTM-sampling/neurips/kl_divergences3.pkl')
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# with open(save_path, 'wb') as f:
#     pickle.dump(kls, f)
# print("saved")

# Save L2 divergences to file
save_path = os.path.expanduser('~/SBTM-sampling/neurips/l2_errors.pkl')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'wb') as f:
    pickle.dump(l2s, f)
print("saved")

#%%
# Load KL divergences from file
save_path = os.path.expanduser('~/SBTM-sampling/neurips/kl_divergences3.pkl')
with open(save_path, 'rb') as f:
    kls = pickle.load(f)
    
# Load L2 errors from file
save_path = os.path.expanduser('~/SBTM-sampling/neurips/l2_errors.pkl')
with open(save_path, 'rb') as f:
    l2s = pickle.load(f)

#%%
"Plot L2 errors"
# Settings
examples = ['gaussians_far', 'gaussians_near']
# method_names = ['sde', 'sbtm', 'svgd']
# annealing_names = ['geometric', 'dilation', 'non-annealed']
method_names = ['sde', 'sbtm']
annealing_names = ['non-annealed']
num_particles_list = [100, 300, 1000, 3000, 10000]

step_size = 0.01
max_steps = 1000
num_particles = 1000
for example_name in ['gaussians_far', 'gaussians_near']:
    plt.figure(figsize=(10/1.2, 6/1.2))
    for method_name in method_names:
        for annealing_name in annealing_names:
            l2_vals = []
            ts = []
            for i in range(max_steps):
                key = (example_name, method_name, annealing_name, num_particles, i)
                if key in l2s:
                    l2_vals.append(l2s[key])
                    ts.append(i * step_size)
            if l2_vals:
                plt.plot(ts, l2_vals, label=f"{method_name}")
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("L2 Error")
    plt.title(f"{example_name}, n={num_particles}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# analytic
example_name = 'analytic'
step_size = 0.002
max_steps = 600

plt.figure(figsize=(10, 6))
for method_name in method_names:
    for annealing_name in annealing_names:
        l2_vals = []
        ts_plot = []
        for i in range(max_steps):
            key = (example_name, method_name, annealing_name, num_particles, i)
            if key in l2s:
                l2_vals.append(l2s[key])
                ts_plot.append(i * step_size)
        if l2_vals:
            plt.plot(ts_plot, l2_vals, label=f"{method_name}")
plt.yscale('log')
plt.xlabel("Time")
plt.ylabel("L2 Error")
plt.title(f"Example {example_name}, n={num_particles}")
plt.legend()
plt.tight_layout()
plt.show()

#%%
"Plot KL divergences"
# Settings
examples = ['gaussians_far', 'gaussians_near']
# method_names = ['sde', 'sbtm', 'svgd']
# annealing_names = ['geometric', 'dilation', 'non-annealed']
method_names = ['sde', 'sbtm']
annealing_names = ['non-annealed']
num_particles_list = [100, 300, 1000, 3000, 10000]

step_size = 0.01
max_steps = 1000
num_particles = 1000
for example_name in ['gaussians_far', 'gaussians_near']:
    plt.figure(figsize=(10/1.2, 6/1.2))
    for method_name in method_names:
        for annealing_name in annealing_names:
            kl_vals = []
            ts = []
            for i in range(max_steps):
                key = (example_name, method_name, annealing_name, num_particles, i)
                if key in kls:
                    kl_vals.append(kls[key])
                    ts.append(i * step_size)
            if kl_vals:
                plt.plot(ts, kl_vals, label=f"{method_name}")
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("KL Divergence")
    plt.title(f"{example_name}, n={num_particles}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# analytic
example_name = 'analytic'
step_size = 0.002
max_steps = 600

def K(t):
    return 1 - jnp.exp(-2 * t)
t0 = 0.1

ts = jnp.arange(max_steps) * step_size
ground_truth_kl = []
for t in ts:
    cov1 = jnp.array([[K(t + t0)]])
    cov2 = jnp.array([[1.0]])
    kl = stats.relative_entropy_gaussians(jnp.zeros(1), cov1, jnp.zeros(1), cov2)
    ground_truth_kl.append(kl)
ground_truth_kl = jnp.array(ground_truth_kl)

plt.figure(figsize=(10, 6))
for method_name in method_names:
    for annealing_name in annealing_names:
        kl_vals = []
        ts_plot = []
        for i in range(max_steps):
            key = (example_name, method_name, annealing_name, num_particles, i)
            if key in kls:
                kl_vals.append(kls[key])
                ts_plot.append(i * step_size)
        if kl_vals:
            plt.plot(ts_plot, kl_vals, label=f"{method_name}")
plt.plot(ts, ground_truth_kl, 'k--', label="Ground Truth", linewidth=2)
plt.yscale('log')
plt.xlabel("Time")
plt.ylabel("KL Divergence")
plt.title(f"Example {example_name}, n={num_particles}")
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Plot KL vs num_particles at last time step
for example_name in ['gaussians_far', 'gaussians_near', 'analytic']:
    if example_name == 'gaussians_far' or example_name == 'gaussians_near':
        step_size = 0.01
        max_steps = 1000
    elif example_name == 'analytic':
        step_size = 0.002
        max_steps = 1250
    plt.figure(figsize=(10, 6))
    for method_name in ['sde', 'sbtm']:
        for annealing_name in annealing_names:
            last_kls = []
            for num_particles in num_particles_list:
                key = (example_name, method_name, annealing_name, num_particles, max_steps-500)
                if key in kls:
                    last_kls.append(kls[key])
            plt.plot(num_particles_list, last_kls, marker='o', label=f"{method_name}-{annealing_name}")
    plt.xlabel("Number of particles")
    plt.ylabel("KL Divergence at final time")
    plt.title(f"Example {example_name}, t={max_steps*step_size}")
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
"Table of errors"
import pandas as pd
examples = ['gaussians_far', 'gaussians_near', 'analytic']
method_names = ['sde', 'sbtm', 'svgd']
annealing_names = ['geometric', 'dilation', 'non-annealed']
num_particles = 100

for example_name in examples:
    if example_name == 'analytic':
        step_size = 0.002
        max_steps = 1250
    else:
        step_size = 0.01
        max_steps = 1000

    data = []
    for method_name in method_names:
        row = []
        for annealing_name in annealing_names:
            key = (example_name, method_name, annealing_name, num_particles, max_steps-1)
            val = kls.get(key, float('nan'))
            row.append(val)
        data.append(row)
    df = pd.DataFrame(data, index=method_names, columns=annealing_names)
    # Format to 3 significant digits
    df = df.map(lambda x: f"{x:.2g}" if pd.notnull(x) else "nan")
    print(f"\nKL at final time for {example_name}:")
    print(df)

#%%
"Table of errors vs num_particles (non-annealed only)"
import pandas as pd
examples = ['gaussians_far', 'gaussians_near', 'analytic']
method_names = ['sde', 'sbtm', 'svgd']
annealing_names = ['non-annealed', 'geometric', 'dilation']
num_particles_list = [100, 300, 1000, 3000, 10000]

for example_name in examples:
    if example_name == 'analytic':
        step_size = 0.002
        max_steps = 1250
        annealing_names = ['non-annealed']
    else:
        step_size = 0.01
        max_steps = 1000

    for annealing_name in annealing_names:
        data = []
        for method_name in method_names:
            row = []
            for num_particles in num_particles_list:
                key = (example_name, method_name, annealing_name, num_particles, max_steps-1)
                val = kls.get(key, float('nan'))
                row.append(val)
            data.append(row)
        df = pd.DataFrame(data, index=method_names, columns=num_particles_list)
        # Format to 3 significant digits
        df = df.map(lambda x: f"{x:.2g}" if pd.notnull(x) else "nan")
        print(f"\nKL at final time for {example_name} ({annealing_name}):")
        print(df)

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
        
#%%
"""1d Analytic solution"""

example_name = 'analytic'
annealing_name = 'non-annealed'
lims = [-5, 5]
for (step_size, max_steps) in tqdm([(0.1, 50), (0.05, 100), (0.02, 250), (0.01, 500), (0.005, 1000), (0.002, 2500)], desc=f'{example_name}'):
    target_samples = target_distributions[example_name].sample(jrandom.PRNGKey(42), size=1000)
    kde = gaussian_kde(target_samples.T)
    for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
        try:
            data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
            path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
            with open(path, 'rb') as f:
                log_data = pickle.load(f)
                
            all_particles = jnp.array([log['particles'] for log in log_data['logs']])

            # initial and final particles
            prior_sample = all_particles[0]
            sample = all_particles[-1]

            # fig, ax = plots.plot_distributions(prior_sample, sample, target_distributions[example_name].density, lims=lims)
            # ax.set_title(fr'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
            # x = jnp.linspace(lims[0], lims[1], 1000)
            # ax.plot(x, kde(x), lw=2, label='Target KDE', color='orange')
            # ax.legend()

            save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
            # plt.savefig(save_path)
            
            # trajectories
            fig, ax = plots.visualize_trajectories(all_particles, particle_idxs = [0,1], max_time=step_size*max_steps, title=f'{method_name} {annealing_name} $\Delta t={step_size}$, $T={max_steps*step_size}$')
            save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}_trajectories.png')
            plt.savefig(save_path)
            
            plt.close('all')
        except:
            print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
            
#%%
"2d Gaussian mixtures"

lims_near = [[-8, 8], [-8, 8]]
lims_far = [[-20, 20], [-20, 20]]
for (step_size, max_steps) in tqdm([(0.01, 10_000)], desc=f'2d Gaussian mixtures'):
# for (step_size, max_steps) in tqdm([(0.01, 10), (0.01, 100), (0.01, 1000), (0.01, 10000), (0.1, 10), (0.1, 100), (0.1, 1000), (0.1, 10000)], desc=f'2d Gaussian mixtures'):
    for (example_name, lims) in tqdm(list(zip(['gaussians_far_2d'], [lims_far, lims_near])), leave=False, desc=f'step_size={step_size}, max_steps={max_steps}'):
        # for annealing_name in tqdm(['dilation', 'geometric', 'non-annealed'], leave=False, desc=f'example={example_name}'):
        for annealing_name in tqdm(['dilation'], leave=False, desc=f'example={example_name}'):
            for method_name in tqdm(['sde', 'sbtm'], leave=False, desc=f'annealing={annealing_name}'):
                try:
                    data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
                    path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
                    with open(path, 'rb') as f:
                        log_data = pickle.load(f)
                    
                    samples = [(i, log_data['logs'][i]['particles']) for i in range(len(log_data['logs'])) if i % ((max_steps-1)//4) == 0]

                    for (i, sample) in samples:
                        fig, ax = plots.plot_distributions_2d(sample, target_distributions[example_name].density, lims=lims)
                        ax.set_title(fr'{method_name} {annealing_name} $\Delta T={step_size}$, $t={i*step_size:.1f}$')

                        save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{max_steps}.png')
                        plt.savefig(save_path)
                        plt.show()
                        plt.close()
                except Exception as e:
                    print(f'Failed for {example_name}, {method_name}, {annealing_name}, {step_size}, {max_steps}')
                    print(e)

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
    t = jnp.clip(t, threshold, 1)
    return target_score(x/t)

def geometric_mean_score(t, x, prior_score, target_score):
    return t * target_score(x) + (1-t) * prior_score(x)

example_name = 'gaussians_far_2d'
d = 2
annealing_name = 'dilation'

step_size = 0.01
max_steps = 10_000

prior_dist = distribution.Gaussian(jnp.zeros(d), jnp.eye(d))

target_dist = target_distributions[example_name]
if annealing_name == 'geometric':
    annealed_score = lambda t,x : geometric_mean_score(λ(t, step_size*max_steps), x, prior_dist.score, target_dist.score)
elif annealing_name == 'dilation':
    annealed_score = lambda t,x : dilation_score(λ(t, step_size*max_steps), x, target_dist.score)
elif annealing_name == 'non-annealed':
    annealed_score = lambda t,x: target_dist.score(x)

#sbtm
data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/sbtm/{annealing_name}')
path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
with open(path, 'rb') as f:
    log_data = pickle.load(f)
particles = jnp.array([log['particles'] for log in log_data['logs']])
scores = jnp.array([log['score'] for log in log_data['logs']])

# sde
# data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/sde/{annealing_name}')
# path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
# with open(path, 'rb') as f:
#     log_data = pickle.load(f)
# particles_sde = jnp.array([log['particles'] for log in log_data['logs']])

# compute d/dt KL
kl_div_time_derivative = -stats.time_derivative(stats.compute_kl_divergences(particles, target_dist.log_density), step_size)
kl_div_time_derivative = jnp.where(jnp.isnan(kl_div_time_derivative), jnp.nanmax(kl_div_time_derivative), kl_div_time_derivative)
kl_div_time_derivative = jnp.clip(kl_div_time_derivative, a_min=1e-5, a_max=1e4)

# kl_div_time_derivative_sde = -stats.time_derivative(stats.compute_kl_divergences(particles_sde, target_dist.log_density), step_size)
# kl_div_time_derivative_sde = jnp.where(jnp.isnan(kl_div_time_derivative_sde), jnp.nanmax(kl_div_time_derivative_sde), kl_div_time_derivative_sde)
# kl_div_time_derivative_sde = jnp.clip(kl_div_time_derivative_sde, a_min=1e-5, a_max=1e4)

# compute relative Fisher info
fisher_divs = []
ts = jnp.linspace(0.0, step_size*max_steps, max_steps)
for ti, particles_i, scores_i in list(zip(ts, particles, scores)):
    value = jnp.mean(jax.vmap(lambda x,y: jnp.dot(x, y))(scores_i - target_dist.score(particles_i), scores_i - annealed_score(ti, particles_i)))
    fisher_divs.append(value)

#%%
steps_to_plot = max_steps
save = False
for smoothing in [0.5]:

    # plot
    fig, ax = plt.subplots(figsize=(10, 2))    
    T = steps_to_plot*step_size
    plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, SBTM', markersize=3, max_time=T)
    plots.plot_quantity_over_time(ax, stats.ema(fisher_divs, smoothing)[:steps_to_plot], label=fr'$\frac{{1}}{{n}}\sum_{{i=1}}^n \langle s(X_i) - \nabla \log \pi(X_i),  s(X_i) - \nabla \log \pi_t(X_i) \rangle$, SBTM', max_time=T)
    # plots.plot_quantity_over_time(ax, stats.ema(kl_div_time_derivative_sde, smoothing)[:steps_to_plot], label=rf'$-\frac{{d}}{{dt}} KL(f_t||\pi)$, SDE', markersize=3, max_time=T)
    ax.set_title(f"{example_name} Δt = {step_size}, T={T} {annealing_name}")

    fig.show()
    if save:
        save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/entropy_dissipation/{annealing_name}')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{steps_to_plot}_particles_20000.png'))
#%%
"2d Gaussian mixtures"

lims_far = [[-20, 20], [-20, 20]]
means=[[-15, -15], [-15, -5], [-15, 5], [-15, 15],
               [-5, -15], [-5, -5], [-5, 5], [-5, 15],
               [5, -15], [5, -5], [5, 5], [5, 15],
               [15, -15], [15, -5], [15, 5], [15, 15]]
step_size, max_steps = 0.01, 10_000
example_name, lims = 'gaussians_far_2d', lims_far
annealing_name = 'dilation'

method_name = 'sbtm'
data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
with open(path, 'rb') as f:
    log_data = pickle.load(f)
samples_sbtm = [(i, log_data['logs'][i]['particles']) for i in range(len(log_data['logs'])) if i % ((max_steps-1)//4) == 0]

method_name = 'sde'
data_dir = os.path.expanduser(f'~/SBTM-sampling/data/{example_name}/{method_name}/{annealing_name}')
path = os.path.join(data_dir, f'stepsize_{step_size}_numsteps_{max_steps}.pkl')
with open(path, 'rb') as f:
    log_data = pickle.load(f)
samples_sde = [(i, log_data['logs'][i]['particles']) for i in range(len(log_data['logs'])) if i % ((max_steps-1)//4) == 0]


#%%
import numpy as np
def plot_distributions_2d(particles, density, lims=None, resolution=400, num_scatter=20000, color=None):
    """Plot the 2D density and scatterplot the transported particles"""
    fig, ax = plt.subplots(figsize=(6, 6))

    if lims is None:
        lims = [
            [particles[:, 0].min()-1, particles[:, 0].max()+1],
            [particles[:, 1].min()-1, particles[:, 1].max()+1]
        ]
    
    # Create a grid of points
    x = np.linspace(lims[0][0], lims[0][1], resolution)
    y = np.linspace(lims[1][0], lims[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    Z = density(positions.T).reshape(X.shape)
    
    ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
    
    scatter_particles = particles if particles.shape[0] <= num_scatter else particles[np.random.choice(particles.shape[0], num_scatter, replace=False)]
    if color is not None:
        # If color is a list/array, subsample it in the same way as particles
        if len(color) == particles.shape[0] and particles.shape[0] > num_scatter:
            color = np.array(color)
            idxs = np.random.choice(particles.shape[0], num_scatter, replace=False)
            color = color[idxs]
        ax.scatter(scatter_particles[:, 0], scatter_particles[:, 1], c=color, s=1, label='Transported Particles')
    else:
        ax.scatter(scatter_particles[:, 0], scatter_particles[:, 1], c='r', s=1, label='Transported Particles')
    
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    return fig, ax

#%%
import matplotlib.colors as mcolors

# Define 16 distinct colors
color_list = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3"
]

means_arr = jnp.array(means)  # shape (16, 2)

# Store figures for later manipulation
sbtm_figs = []
sde_figs = []

method_name = 'sbtm'
# Compute final assignment of each particle to closest mean
final_particles = samples_sbtm[-1][1]  # shape (num_particles, 2)
dists = jnp.linalg.norm(final_particles[:, None, :] - means_arr[None, :, :], axis=-1)  # (num_particles, 16)
assignments = jnp.argmin(dists, axis=1)  # (num_particles,)

for (i, sample) in samples_sbtm:
    # Assign each particle to the closest mean (using final assignment)
    colors = [color_list[assignments[j]] for j in range(sample.shape[0])]
    fig, ax = plot_distributions_2d(
        sample, target_distributions[example_name].density, lims=lims, color=colors
    )
    ax.set_title(fr'{method_name} $t={i*step_size:.1f}$')

    save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{i+1}.png')
    plt.savefig(save_path)
    sbtm_figs.append(fig)
    plt.show()
    plt.close()

method_name = 'sde'
# Compute final assignment of each particle to closest mean
final_particles = samples_sde[-1][1]  # shape (num_particles, 2)
dists = jnp.linalg.norm(final_particles[:, None, :] - means_arr[None, :, :], axis=-1)  # (num_particles, 16)
assignments = jnp.argmin(dists, axis=1)  # (num_particles,)

for (i, sample) in samples_sde:
    # Assign each particle to the closest mean (using final assignment)
    colors = [color_list[assignments[j]] for j in range(sample.shape[0])]
    fig, ax = plot_distributions_2d(
        sample, target_distributions[example_name].density, lims=lims, color=colors
    )
    ax.set_title(fr'{method_name} $t={i*step_size:.1f}$')

    save_dir = os.path.expanduser(f'~/SBTM-sampling/plots/{example_name}/{method_name}/{annealing_name}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'stepsize_{step_size}_numsteps_{i+1}.png')
    plt.savefig(save_path)
    sde_figs.append(fig)
    plt.show()
    plt.close()

