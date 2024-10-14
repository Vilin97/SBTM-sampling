import jax.numpy as jnp
import jax
from typing import Callable

def divergence(f: Callable, mode, n=1000):

    """
    Compute the divergence of a vector field using JAX.
    
    Args:
    f : Callable
        The vector field function R^n -> R^n.
    n : int
        Mode of divergence computation. -1 for exact trace, 0 for efficient exact, 
        and positive integers for stochastic estimation using Hutchinson's trace estimator.
    Returns:
    Callable
        A function that computes the divergence at a point.
    """
    assert mode in ['forward', 'reverse', 'optimized', 'approximate_gaussian', 'approximate_rademacher', 'denoised'], "Invalid mode"
    
    if mode == 'forward':
        return jax.jit(lambda x: jnp.trace(jax.jacfwd(f)(x)))
    if mode == 'reverse':
        return jax.jit(lambda x: jnp.trace(jax.jacrev(f)(x)))
    if mode == 'optimized':
        def div(x):
            fi = lambda i, *y: f(jnp.stack(y))[i]
            dfidxi = lambda i, y: jax.grad(fi, argnums=i+1)(i, *y)
            return sum(dfidxi(i, x) for i in range(x.shape[0]))
        return jax.jit(div)
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


def mean_divergence(f, particles, mode, n=10):
    if mode in ['forward', 'reverse', 'optimized']:
        return jnp.mean(jax.vmap(divergence(f, mode, n))(particles))
    else:
        key = jax.random.split(jax.random.key(42), particles.shape[0])
        return jnp.mean(jax.vmap(divergence(f, mode, n))(particles, key))

forward_mean_divergence = lambda f, particles: mean_divergence(f, particles, 'forward')
reverse_mean_divergence = lambda f, particles: mean_divergence(f, particles, 'reverse')
optimized_mean_divergence = lambda f, particles: mean_divergence(f, particles, 'optimized')
denoised_mean_divergence = lambda f, particles: mean_divergence(f, particles, 'denoised')
approximate_gaussian_mean_divergence = lambda f, particles: mean_divergence(f, particles, 'approximate_gaussian')
approximate_rademacher_mean_divergence = lambda f, particles: mean_divergence(f, particles, 'approximate_rademacher')



from flax import nnx

class MLP(nnx.Module):
    def __init__(self, d, num_hidden, rngs: nnx.Rngs):
        self.linear = nnx.Linear(d, num_hidden, rngs=rngs)
        self.linear_out = nnx.Linear(num_hidden, d, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear(x))
        x = self.linear_out(x)
        return x
    
import time
def timeit(func, *args):
    start = time.time()
    k = 10
    for _ in range(k):
        func(*args).block_until_ready()
    return (time.time() - start) / k * 10**6

d = 10
f = MLP(d, 100, rngs=nnx.Rngs(0))
key = jax.random.PRNGKey(1)
x = jax.random.normal(key, (d,))
key, subkey = jax.random.split(key)
f(x)

forward_divergence = divergence(f, 'forward')
reverse_divergence = divergence(f, 'reverse')
optimized_divergence = divergence(f, 'optimized')
denoised_divergence = divergence(f, 'denoised', n=1000)
approximate_gaussian_divergence = divergence(f, 'approximate_gaussian', n=1000)
approximate_rademacher_divergence = divergence(f, 'approximate_rademacher', n=1000)

# correctness
forward_divergence(x)
reverse_divergence(x)
optimized_divergence(x)
denoised_divergence(x, key)
approximate_gaussian_divergence(x, key)
approximate_rademacher_divergence(x, key)

# timing
timeit(forward_divergence, x)
timeit(reverse_divergence, x)
timeit(optimized_divergence, x)
timeit(denoised_divergence, x, key)
timeit(approximate_gaussian_divergence, x, key)
timeit(approximate_rademacher_divergence, x, key)

# particle ensemble timing
n = 1000
particles = jax.random.normal(key, (n, d))
timeit(jax.vmap(forward_divergence), particles)
timeit(jax.vmap(reverse_divergence), particles)
timeit(jax.vmap(optimized_divergence), particles)
timeit(jax.vmap(denoised_divergence), particles, jax.random.split(key, n))
timeit(jax.vmap(approximate_gaussian_divergence), particles, jax.random.split(key, n))
timeit(jax.vmap(approximate_rademacher_divergence), particles, jax.random.split(key, n))

forward_mean_divergence(f, particles)
reverse_mean_divergence(f, particles)
denoised_mean_divergence(f, particles)
approximate_gaussian_mean_divergence(f, particles)
approximate_rademacher_mean_divergence(f, particles)

timeit(forward_mean_divergence, f, particles)
timeit(reverse_mean_divergence, f, particles)
timeit(denoised_mean_divergence, f, particles)
timeit(approximate_gaussian_mean_divergence, f, particles)
timeit(approximate_rademacher_mean_divergence, f, particles)

# taking NN gradient
import optax
def update(loss):
    grad_fn = nnx.value_and_grad(loss)
    value, grads = grad_fn(f, particles)
    optimizer = nnx.Optimizer(f, optax.adamw(0.005, 0.9))
    optimizer.update(grads)
    return value

forward_mean_divergence(f, particles)
timeit(update, forward_mean_divergence) # 2nd fastest
# forward_mean_divergence(f, particles)
timeit(update, reverse_mean_divergence) # fastest
# forward_mean_divergence(f, particles)
timeit(update, denoised_mean_divergence)
# forward_mean_divergence(f, particles)
timeit(update, approximate_gaussian_mean_divergence)
# forward_mean_divergence(f, particles)
timeit(update, approximate_rademacher_mean_divergence)
forward_mean_divergence(f, particles)



