import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax.scipy.stats import gaussian_kde
from tqdm import tqdm
from sbtm import stats

def plot_distributions(initial_particles, transported_particles, density, lims=None):
    """Plot the initial and transported particles, and the target density function, in 1D"""
    fig, ax = plt.subplots(figsize=(10, 6))
    if lims:
        x = np.linspace(lims[0], lims[1], 1000)
        transported_particles = transported_particles[jnp.logical_and(transported_particles >= lims[0], transported_particles <= lims[1])]
    else:
        x = np.linspace(min(initial_particles + transported_particles) - 1, max(initial_particles + transported_particles) + 1, 1000)


    # Plot initial particles
    ax.hist(initial_particles, bins=30, density=True, alpha=0.4, color='b', histtype='bar', label='Initial Particles')
    kde_initial = gaussian_kde(initial_particles.T)(jnp.reshape(x, (1, -1)))
    ax.plot(x, kde_initial, 'b-', lw=2, label='Initial KDE')

    # Plot histogram of transported particles
    ax.hist(transported_particles, bins=30, density=True, alpha=0.4,
            color='g', histtype='bar', label='Transported Particles')
    kde_transported = gaussian_kde(transported_particles.T)(jnp.reshape(x, (1, -1)))
    ax.plot(x, kde_transported, 'g-', lw=2, label='Transported KDE')

    # Plot the target density function
    y = density(jnp.reshape(x, (-1, 1))) # reshape to (n, 1) for broadcasting
    ax.plot(x, y, 'r-', lw=2, label='Target Distribution')

    ax.set_xlabel('Particle Value', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    return fig, ax

def plot_distributions_2d(transported_particles, density, lims=None):
    """Plot the 2D density and scatterplot the transported particles"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a grid of points
    x_min = lims[0][0] if lims else min(transported_particles[:, 0])
    x_max = lims[0][1] if lims else max(transported_particles[:, 0])
    y_min = lims[1][0] if lims else min(transported_particles[:, 1])
    y_max = lims[1][1] if lims else max(transported_particles[:, 1])
    
    x = np.linspace(x_min - 1, x_max + 1, 100)
    y = np.linspace(y_min - 1, y_max + 1, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Evaluate the density on the grid
    Z = density(positions.T).reshape(X.shape)
    
    # Plot the density as a heatmap
    ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
    
    # Scatterplot the transported particles
    ax.scatter(transported_particles[:, 0], transported_particles[:, 1], c='r', s=1, label='Transported Particles')
    
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    return fig, ax

def visualize_trajectories(particles, particle_idxs = [0,1,2,3,4], max_time=None, title=None):
    """Make a heatmap of particles over time and overlay trajectories of a few particles"""
    def kde(x_values, particles):
        density_values = []
        for particles_i in tqdm([particles_i[:,0] for particles_i in particles], desc="Computing KDE"):
            kde = gaussian_kde(particles_i)
            density_values.append(kde(x_values))
        return density_values

    def plot_density_evolution(x_values, density_values, title, trajectories):
        assert len(x_values) == len(density_values[0])
        xmin, xmax = x_values[0], x_values[-1]
        num_x_values = len(x_values)
        num_iterations = len(density_values)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(jnp.array(density_values).T)
        plt.yticks(ticks=jnp.linspace(0, num_x_values, 9), labels=[f'{int(x)}' for x in jnp.linspace(xmin, xmax, 9)], rotation=0)
        if max_time is not None:
            plt.xticks(ticks=jnp.linspace(0, num_iterations, 9), 
                   labels=jnp.linspace(0, num_iterations, 9) * max_time / num_iterations, 
                   rotation=0)
            plt.xlabel('Time')
        else:
            plt.xticks(ticks=jnp.linspace(0, num_iterations, 9), 
                   labels=[f'{int(x)}' for x in jnp.linspace(0, num_iterations, 9)], 
                   rotation=0)
            plt.xlabel('Iteration')
        plt.title(title)
        
        for trajectory in trajectories:
            trajectory_mapped = [jnp.argmin(jnp.abs(x_values - value)) for value in trajectory]
            plt.plot(jnp.linspace(0, len(density_values), len(trajectory)), trajectory_mapped, color='white', marker='o', markersize=2)
        plt.show()
    
    x_values = jnp.linspace(-10, 10, 200)
    sde_kde = kde(x_values, particles)
    trajectories = [[particles_i[j,0] for particles_i in particles] for j in particle_idxs]
    plot_density_evolution(x_values, sde_kde, title, trajectories)
    
def plot_quantity_over_time(ax, quantity, label, yscale='linear', plot_zero_line=True, max_time=None, **kwargs):
    """Plot an arbitrary quantity over time"""
    ax.plot(quantity, label=label, **kwargs)
    if yscale == 'linear' and plot_zero_line:
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    ax.set_yscale(yscale)
    ax.legend()
    if max_time is not None:
        ax.set_xticks(np.linspace(0, len(quantity) - 1, 9))
        ax.set_xticklabels(np.linspace(0, max_time, 9))
        ax.set_xlabel('Time')
    else:
        ax.set_xlabel('Step')
    return ax

def plot_kl_divergence(particles, target_density, **kwargs):
    """Compute and plot the KL divergence between the particles and the target density"""
    kl_divergences = stats.compute_kl_divergences(particles, target_density)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_quantity_over_time(ax, kl_divergences, label='KL Divergence', **kwargs)
    ax.set_ylabel(r'$\frac{1}{n} \sum_{i} \log\left(\frac{f(x_{i})}{π(x_{i})}\right)$')
    plt.title('KL Divergence over Time')
    plt.show()

def plot_fisher_divergence(particles, target_score, scores=None, **kwargs):
    """"Plot the Fisher divergence over time
            1/n ∑ᵢ |∇log f(xᵢ) - ∇log π(xᵢ)|² """
    if scores is None:
        scores = [stats.compute_score(sample_f) for sample_f in particles]
    fisher_divs = stats.compute_fisher_divergences(particles, scores, target_score)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_quantity_over_time(ax, fisher_divs, label='Fisher Divergence', **kwargs)
    ax.set_ylabel(r'$\frac{1}{n} \sum_{i} ||\nabla \log f_t(x_{i}) - \nabla \log \pi(x_{i})||^2$')
    plt.title('Fisher Divergence Estimate')
    plt.show()

### SBTM ###
def plot_losses(loss_values, batch_loss_values, **kwargs):
    """Plot training loss of the score model in SBTM"""
    
    ema_batch_losses = stats.exponential_moving_average(batch_loss_values, smoothing=0.95)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    plot_quantity_over_time(axs[0], loss_values, label='Losses', **kwargs)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel(r'$\frac{1}{n} \sum_{i} ||s(x_{i})||^2 + 2 \nabla \cdot s(x_{i})$')
    axs[0].set_title(r'Implicit loss through iterations')

    plot_quantity_over_time(axs[1], batch_loss_values, label='Batch Losses', **kwargs)
    plot_quantity_over_time(axs[1], ema_batch_losses, label='Exponential Moving Average', **kwargs)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel(r'Batch Loss')
    axs[1].set_title(r'Batch loss through iterations')

    plt.tight_layout()
    plt.show()
    
    
