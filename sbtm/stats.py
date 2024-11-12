import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from tqdm import tqdm

def kl_divergence(sample_f, g):
    """∫ log(f/g) df ≈ 1/n ∑ᵢ log(f(xᵢ) / g(xᵢ)) where f is estimated with KDE"""
    f_kde = gaussian_kde(sample_f.T)
    return jnp.clip(jnp.mean(jnp.log(f_kde(sample_f.T) / g(sample_f))), a_min=0, a_max=None)


def compute_kl_divergences(particles, target_density):
    """Compute the KL divergence between the particles and the target density, every k steps"""

    kl_divergences = []
    for i, particles_i in enumerate(tqdm(particles, desc="Computing KL divergence")):
        kl_div = kl_divergence(particles_i, target_density)
        if kl_div == jnp.inf:
            kl_div = kl_divergences[-1]
        kl_divergences.append(kl_div)

    return kl_divergences

def compute_score(sample_f):
    """Use kde to estimate the score at each particle location"""
    f_kde = gaussian_kde(sample_f.T)
    log_density = lambda x: jnp.log(f_kde(x))[0]
    score_fun = jax.grad(log_density)
    return jax.vmap(score_fun)(sample_f)

def compute_fisher_divergences(particles, scores, target_score):
    """
    Compute the Fisher divergences for a set of particles and their corresponding scores.
    
    Parameters:
    particles (list or array-like): A collection of particle arrays.
    scores (list or array-like): A collection of score arrays at particle locations.
    target_score (callable): A function that computes the target score for a given set of particles.
    
    Returns:
    list: A list of Fisher divergence values for each set of particles and scores.
    """
    
    fisher_divs = []
    for particles_i, scores_i in tqdm(list(zip(particles, scores)), desc="Computing Fisher divergence"):
        value = jnp.mean(jax.vmap(square_norm_diff)(scores_i, target_score(particles_i)))
        fisher_divs.append(value)

    return fisher_divs

def ema(data, smoothing):
    """Smooothing = 0 means no smoothing, smoothing = 1 means no change"""
    return exponential_moving_average(data, smoothing)

def exponential_moving_average(data, smoothing):
    """Smooothing = 0 means no smoothing, smoothing = 1 means no change"""
    ema = []
    ema_current = data[0]
    for value in data:
        ema_current = (1 - smoothing) * value + smoothing * ema_current
        ema.append(ema_current)
    return ema


@jax.jit
def square_norm_diff(x, y):
    """|x - y|²"""
    return jnp.sum(jnp.square(x - y))
