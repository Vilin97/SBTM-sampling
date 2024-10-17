import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm

def plot_distributions(initial_particles, transported_particles, density_obj):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of initial particles
    ax.hist(initial_particles, bins=30, density=True, alpha=0.4, color='b',
            histtype='bar', label='Initial Particles')

    # Plot histogram of transported particles
    ax.hist(transported_particles, bins=30, density=True, alpha=0.4,
            color='g', histtype='bar', label='Transported Particles')

    # Plot the target density function
    x = np.linspace(min(initial_particles + transported_particles) - 1, 
                    max(initial_particles + transported_particles) + 1, 1000)
    y = jax.vmap(density_obj.density)(x)
    ax.plot(x, y, 'r-', lw=2, label='Target Distribution')

    ax.set_xlabel('Particle Value', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    return fig, ax

def plot_losses(loss_values, batch_loss_values):
    """Plot training loss of the score model in SBTM"""
    def exponential_moving_average(data, smoothing):
        ema = []
        ema_current = data[0]
        for value in data:
            ema_current = (1 - smoothing) * value + smoothing * ema_current
            ema.append(ema_current)
        return ema

    ema_batch_losses = exponential_moving_average(batch_loss_values, smoothing=0.95)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(loss_values, label='Losses')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\frac{1}{n} \sum_{i} ||s(x_{i})||^2 + 2 \nabla \cdot s(x_{i})$')
    plt.title(r'Implicit loss through iterations')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(batch_loss_values, label='Batch Losses')
    plt.plot(ema_batch_losses, label='Exponential Moving Average', color='red')
    plt.xlabel('Iteration')
    plt.ylabel(r'Batch Loss')
    plt.title(r'Batch loss through iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
def plot_fisher_divergence(particles, scores, target_score):
    """"Plot the Fisher divergence over time
            1/n ∑ᵢ |s(xᵢ) - ∇log π(xᵢ)|² """
    @jax.jit
    def fisher_divergence(score_values_1, score_values_2):
        """ |∇log f₁(x) - ∇log f₂(x)|² """
        return jnp.sum(jnp.square(score_values_1 - score_values_2))
    
    fisher_divs = []
    for (particles_i, scores_i) in tqdm(list(zip(particles, scores)), desc="Computing Fisher divergence"):
        value = jnp.mean(jax.vmap(fisher_divergence)(scores_i, target_score(particles_i)))
        fisher_divs.append(value)
        
    plt.figure(figsize=(6, 6))
    plt.plot(fisher_divs)
    plt.title('Fisher Divergence Estimate')
    plt.xlabel('Step')
    plt.ylabel(r'$\frac{1}{n} \sum_{i} ||s(x_{i}) - \nabla \log \pi(x_{i})||^2$')
    plt.show()


def visualize_trajectories(particles, title, particle_idxs = [0,1,2,3,4]):
    """Make a heatmap of particles over time and overlay trajectories of a few particles"""
    def kde(x_values, particles):
        density_values = []
        for particles_i in tqdm([particles_i[:,0] for particles_i in particles], desc="Computing KDE"):
            kde = gaussian_kde(particles_i)
            density_values.append(kde(x_values))
        return density_values

    def plot_density_evolution(x_values, density_values, title, trajectories):
        assert len(x_values) == len(density_values[0])
        assert len(density_values) == len(trajectories[0])
        xmin, xmax = x_values[0], x_values[-1]
        num_x_values = len(x_values)
        num_iterations = len(density_values)
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(jnp.array(density_values)[::-1,:])
        plt.xticks(ticks=jnp.linspace(0, num_x_values, 9), labels=[f'{int(x)}' for x in jnp.linspace(xmin, xmax, 9)], rotation=0)
        plt.yticks(ticks=jnp.linspace(0, num_iterations, 9), labels=[f'{int(x)}' for x in jnp.linspace(num_iterations, 0, 9)], rotation=0)
        plt.ylabel('Iteration')
        plt.title(title)
        
        for trajectory in trajectories:
            trajectory_mapped = [jnp.argmin(jnp.abs(x_values - value)) for value in trajectory[::-1]]
            plt.plot(trajectory_mapped, jnp.linspace(0, len(density_values), len(trajectory)), color='white', marker='o', markersize=2)
        plt.show()
    
    x_values = jnp.linspace(-10, 10, 200)
    sde_kde = kde(x_values, particles)
    trajectories = [[particles_i[j,0] for particles_i in particles] for j in particle_idxs]
    plot_density_evolution(x_values, sde_kde, title, trajectories)
    
    
def plot_kl_divergence(particles, target_density):
    """Plot the KL divergence between the particles and the target density, every k steps"""
    def kl_divergence(sample_f, g):
        """ ∫ log(f/g) df ≈ 1/n ∑ᵢ log(f(xᵢ) / g(xᵢ)) where f is estimated with KDE """
        f_kde = gaussian_kde(sample_f.T)
        return jnp.clip(jnp.mean(jnp.log(f_kde(sample_f.T) / g(sample_f))), a_min=0, a_max=None)

    kl_divergences = []
    for i, particles_i in enumerate(tqdm(particles, desc="Computing KL divergence")):
        kl_div = kl_divergence(particles_i, target_density)
        if kl_div == jnp.inf:
            kl_div = kl_divergences[-1]
        kl_divergences.append(kl_div)

    plt.figure(figsize=(6, 6))
    plt.plot(kl_divergences)
    plt.axhline(y=0, color='grey', linestyle='--', linewidth=0.5, label='KL = 0')
    plt.legend()
    plt.title('KL Divergence Over Time')
    plt.xlabel('Step')
    plt.ylabel(r'$\frac{1}{n} \sum_{i} \log\left(\frac{f(x_{i})}{g(x_{i})}\right)$')
    plt.show()
