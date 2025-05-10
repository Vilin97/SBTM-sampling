import jax
import jax.numpy as jnp
from typing import Callable
from flax import nnx

def divergence(f: Callable, mode, n=100):

    """
    Compute the divergence of a vector field using JAX.

    forward and reverse modes are the fastest and are exact, approximate_rademacher is the fastest and most accurate stochastic estimator.
    
    Args:
    f : Callable
        Vector field.
    mode : str
        Mode of divergence computation.
    n : int
        Number of samples for stochastic estimation.
    
    Returns:
    Callable
        A function that computes the divergence at a point.
    """
    assert mode in ['forward', 'reverse', 'approximate_gaussian', 'approximate_rademacher', 'denoised'], "Invalid mode"
    
    if mode == 'forward':
        return jax.jit(lambda x: jnp.trace(jax.jacfwd(f)(x)))
    if mode == 'reverse':
        return jax.jit(lambda x: jnp.trace(jax.jacrev(f)(x)))
    if mode == 'denoised':
        alpha = jnp.float32(0.1)
        def div(x, key):
            def denoise(key):
                v = jax.random.normal(key, x.shape, dtype=x.dtype)
                return jnp.dot(f(x + alpha * v) - f(x - alpha * v), v) / alpha
            return jax.vmap(denoise)(jax.random.split(key, n)).mean()
        return jax.jit(div)
    else:
        def div(x, key):
            def vJv(key):
                _, vjp = jax.vjp(f, x)
                f_rand = jax.random.normal if mode == 'approximate_gaussian' else jax.random.rademacher
                v = f_rand(key, x.shape, dtype=x.dtype)
                return jnp.dot(vjp(v)[0], v)
            return jax.vmap(vJv)(jax.random.split(key, n)).mean()
        return jax.jit(div)

@nnx.jit
def explicit_score_matching_loss(s, particles, target_score_values):
    """
    Compute the score matching loss between the vector field s and the true score.
    1/n ∑ᵢ ||s(xᵢ) - ∇log f*(xᵢ)||²
    """
    return jnp.sum(jnp.square(s(particles) - target_score_values)) / particles.shape[0]

@nnx.jit
def weighted_explicit_score_matching_loss(s, particles, target_score_values, weighting):
    """
    Compute the weighted score matching loss between the vector field s and the true score.
    1/n ∑ᵢ ⟨s(xᵢ) - ∇log f*(xᵢ), D[i] (s(xᵢ) - ∇log f*(xᵢ))⟩
    """
    def weighted_loss(x, target, D):
        diff = s(x) - target
        return jnp.dot(diff, jnp.dot(D, diff))
    return jnp.mean(jax.vmap(weighted_loss)(particles, target_score_values, weighting))

@nnx.jit
def implicit_score_matching_loss(s, particles):
    """
    Compute the implicit score matching loss for vector field s
    1/n ∑ᵢ ||s(xᵢ)||^2 + 2 ∇⋅s(xᵢ)
    """
    div = divergence(s, 'reverse')
    def loss(x):
        return jnp.sum(jnp.square(s(x))) + 2 * div(x)
    return jnp.mean(jax.vmap(loss)(particles))
