import jax.numpy as jnp
import jax
from jax import grad, vmap

def s(x):
    # Define your function s here
    return x  # Example: identity function

def divergence_s(x):
    # Compute the divergence of s at x
    return jnp.trace(jax.jacfwd(s)(x))

def compute_expression(x):
    s_x = s(x)
    s_x_squared = jnp.sum(s_x**2)
    div_s_x = divergence_s(x)
    return s_x_squared + 2 * div_s_x

def compute_sum(x):
    # Vectorize the computation over all rows
    compute_expression_vmap = vmap(compute_expression)
    return jnp.sum(compute_expression_vmap(x))

# Example usage
n, d = 5, 3  # Example dimensions
x = jnp.ones((n, d))  # Example input array
result = compute_sum(x)
print(result)

# TODO: compare with divergence implementation in https://github.com/jax-ml/jax/issues/3022#issuecomment-2100553108