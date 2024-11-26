"""Compare D-weighted and unweighted loss functions for score matching."""

#%%
from jax import grad, jit
import jax

import jax.numpy as jnp

# Constants
# B = 1 / 24
B = 1
d = 3

def A(z):
    z_abs_sq = jnp.linalg.norm(z)**2
    delta = jnp.eye(len(z))
    return B * (z_abs_sq * delta - jnp.outer(z, z))

def K(t):
    return 1 - jnp.exp(-2 * B * (d - 1) * t)

def P(K):
    return ((d + 2) * K - d) / (2 * K)

def Q(K):
    return (1 - K) / (2 * K**2)

def ustar_t(x, t):
    K_t = K(t)
    P_t = P(K_t)
    Q_t = Q(K_t)
    norm_factor = (2 * jnp.pi * K_t)**(-d / 2)
    exp_factor = jnp.exp(-jnp.linalg.norm(x)**2 / (2 * K_t))
    return norm_factor * exp_factor * (P_t + Q_t * jnp.linalg.norm(x)**2)

# Example usage
z = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([1.0, 2.0, 3.0])
t = 5.5

A_result = A(z)
ustar_result = ustar_t(x, t)

print("A(z):", A_result)
print("ustar_t(x, t):", ustar_result)

# Eigenvalues of A_result
eigvals_A, _ = jnp.linalg.eigh(A_result)
print("Eigenvalues of A(z):", eigvals_A)

#%%
import matplotlib.pyplot as plt

x_values = jnp.linspace(-5, 5, 100)
t = 5.5
ustar_values = [ustar_t(jnp.array([x, 0, 0]), t) for x in x_values]

plt.plot(x_values, ustar_values)
plt.xlabel('x')
plt.ylabel('ustar_t([x, 0, 0], t)')
plt.title('Plot of ustar_t([x, 0, 0], t) for x in range -5 to 5')
plt.grid(True)
plt.show()

#%%
def D(x, X):
    n, d = X.shape
    A_sum = jnp.sum(jnp.array([A(x - X[j]) for j in range(n)]), axis=0)
    return A_sum / n

# Example usage
X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
x = jnp.array([1.0, 2.0, 3.0])

D_result = D(x, X)
print("D(x, X):", D_result)

# Eigen decomposition of D_result
eigvals, eigvecs = jnp.linalg.eigh(D_result)

print("Eigenvalues of D(x, X):", eigvals)

#%%
def loss_function(X, s, D):
    
    def log_ustar_grad(x):
        return grad(lambda x: jnp.log(ustar_t(x, t)))(x)
    
    log_ustar_grads = jax.vmap(log_ustar_grad)(X)
    s_values = jax.vmap(s)(X)
    
    diff = log_ustar_grads - s_values
    diff_D = jax.vmap(lambda x, Di: jnp.dot(x.T, jnp.dot(Di, x)))(diff, D)
    
    return jnp.mean(diff_D)

# Example usage
s = lambda x: x  # Replace with the actual function s
X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
D_precomputed = jnp.array([D(x, X) for x in X])
D_precomputed


# TODO: this gives a NAN. 
# loss = loss_function(X, s, D_precomputed)
# print("Loss:", loss)