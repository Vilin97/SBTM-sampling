import numpy as np
import jax
import matplotlib.pyplot as plt

def plot_distributions(initial_particles, transported_particles, density_obj):
    fig = plt.figure(figsize=(10, 6))

    # Plot histogram of initial particles
    plt.hist(initial_particles, bins=30, density=True, alpha=0.4, color='b',
             histtype='bar', label='Initial Particles')

    # Plot histogram of transported particles
    plt.hist(transported_particles, bins=30, density=True, alpha=0.4,
             color='g', histtype='bar', label='Transported Particles')

    # Plot the target density function
    x = np.linspace(min(initial_particles + transported_particles) - 1, 
                    max(initial_particles + transported_particles) + 1, 1000)
    y = jax.vmap(density_obj.density)(x)
    plt.plot(x, y, 'r-', lw=2, label='Target Distribution')

    # plt.title('Initial and Final Distributions of Particles')
    plt.xlabel('Particle Value', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return fig
