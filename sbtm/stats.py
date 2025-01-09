import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from tqdm import tqdm
from flax import nnx

@nnx.jit(static_argnames='log_g')
def kl_divergence_log_density(sample_f, log_g):
    """∫ log(f/g) df ≈ 1/n ∑ᵢ log(f(xᵢ)) - log(g(xᵢ)) where f is estimated with KDE"""
    f_kde = gaussian_kde(sample_f.T)
    return jnp.mean(jnp.log(f_kde(sample_f.T)) - log_g(sample_f))

def kl_divergence(sample_f, g):
    log_g = lambda x: jnp.log(g(x))[0]
    return kl_divergence_log_density(sample_f, log_g)

def compute_kl_divergences(particles, log_density):
    return [kl_divergence_log_density(particles_i, log_density) for particles_i in particles]

def time_derivative(quantity, step_size):
    return jnp.diff(jnp.array(quantity)) / step_size

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
    for particles_i, scores_i in list(zip(particles, scores)):
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

def relative_entropy_gaussians(mean1, cov1, mean2, cov2):
    dim = mean1.shape[0]
    cov2_inv = jnp.linalg.inv(cov2)
    mean_diff = mean2 - mean1
    term1 = jnp.trace(cov2_inv @ cov1)
    term2 = mean_diff.T @ cov2_inv @ mean_diff
    term3 = -dim
    term4 = jnp.log(jnp.linalg.det(cov2) / jnp.linalg.det(cov1))
    return 0.5 * (term1 + term2 + term3 + term4)