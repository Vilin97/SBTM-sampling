"""Compare D-weighted and unweighted loss functions for score matching."""

#%%
from jax import grad, jit
import jax
import numpy as np
from scipy.integrate import quad
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Constants
B = 1 / 24
d = 3

#%%
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
    A_mean = jnp.mean(jax.vmap(lambda row: A(x - row))(X), axis=0)
    return A_mean

# Example usage
X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
x = jnp.array([1.0, 2.0, 3.0])

D_result = D(x, X)
print("D(x, X):", D_result)

# Eigen decomposition of D_result
eigvals, eigvecs = jnp.linalg.eigh(D_result)

print("Eigenvalues of D(x, X):", eigvals)

#%%
def sample_ustar_t(n, t):
    samples = []
    K_t = K(t)
    envelope = lambda x: 1 / jnp.sqrt((2 * jnp.pi)**d * K_t * 1.5) * jnp.exp(-0.5 * jnp.linalg.norm(x)**2 / (K_t * 1.5))
    M = max(ustar_t(jnp.array([x,0,0]), t) / envelope(jnp.array([x,0,0])) for x in np.linspace(-10, 10, 1000)) + 0.1
    
    rejection_steps = 0
    while len(samples) < n:
        x_candidate = np.random.normal(scale=jnp.sqrt(K(t) * 1.5), size=d)
        ustar_value = ustar_t(x_candidate, t)
        g_value = envelope(x_candidate)
        if M * g_value < ustar_value:
            raise ValueError(f"M = {M} is too low: {M * g_value} = Mg(x) < f(x) = {ustar_value} for x = {x_candidate}.")
        if np.random.rand() * M * g_value < ustar_value:
            samples.append(x_candidate)
        else:
            rejection_steps += 1
    
    total_steps = len(samples) + rejection_steps
    rejection_fraction = rejection_steps / total_steps
    print(f"Number of rejection steps: {rejection_steps}")
    print(f"Fraction of rejection steps: {rejection_fraction:.4f}")
    
    return jnp.array(samples)

# Example usage
n_samples = 100
t = 5.5
samples = sample_ustar_t(n_samples, t)
print("Sampled points:", samples)

# Plot histogram of sampled points and the ustar_t pdf
plt.figure(figsize=(10, 6))

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
X = samples
D_precomputed = jnp.array([D(x, X) for x in X])

loss = loss_function(X, s, D_precomputed)
print("Loss:", loss)

# Compute loss using identity matrix instead of D
I = jnp.eye(d)
D_identity = jnp.array([I for _ in X])

loss_identity = loss_function(X, s, D_identity)
print("Loss with identity matrix:", loss_identity)