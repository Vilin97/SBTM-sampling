"""Compare D-weighted and unweighted loss functions for score matching."""

#%%
from jax import grad, jit
import jax
import numpy as np
from scipy.integrate import quad
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sbtm import models
from flax import nnx
from tqdm import tqdm
import optax

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

@jit
def ustar_t(x, t):
    K_t = K(t)
    P_t = P(K_t)
    Q_t = Q(K_t)
    norm_factor = (2 * jnp.pi * K_t)**(-d / 2)
    exp_factor = jnp.exp(-jnp.linalg.norm(x)**2 / (2 * K_t))
    return norm_factor * exp_factor * (P_t + Q_t * jnp.linalg.norm(x)**2)

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
def rejection_sample(n, t, f):
    samples = []
    K_t = K(t)
    proposal = lambda x: 1 / jnp.sqrt((2 * jnp.pi)**d * K_t * 1.5) * jnp.exp(-0.5 * jnp.linalg.norm(x)**2 / (K_t * 1.5))
    x_values = jnp.linspace(-10, 10, 1000)
    x_array = jnp.stack([x_values, jnp.zeros_like(x_values), jnp.zeros_like(x_values)], axis=-1)
    ustar_values = jax.vmap(f)(x_array)
    proposal_values = jax.vmap(proposal)(x_array)
    M = jnp.max(ustar_values / proposal_values) + 0.1
    print(f"Using M = {M}")
    
    rejection_steps = 0
    while len(samples) < n:
        x_candidate = np.random.normal(scale=jnp.sqrt(K(t) * 1.5), size=d)
        ustar_value = f(x_candidate)
        g_value = proposal(x_candidate)
        if M * g_value < ustar_value:
            raise ValueError(f"M = {M} is too low: {M * g_value} = Mg(x) < f(x) = {ustar_value} for x = {x_candidate}.")
        if np.random.rand() * M * g_value < ustar_value:
            samples.append(x_candidate)
        else:
            rejection_steps += 1
    
    total_steps = len(samples) + rejection_steps
    rejection_fraction = rejection_steps / total_steps
    print(f"Number of rejected steps: {rejection_steps}")
    print(f"Fraction of rejected steps: {rejection_fraction:.4f}")
    
    return jnp.array(samples)

#%%
from sbtm import density

t = 5.5
ustar = lambda x: ustar_t(x, t)
x_values = jnp.linspace(-5, 5, 200)
ustar_values = [ustar(jnp.array([x, 0, 0])) for x in x_values]
target_score = lambda x: density.score(ustar, x)

plt.plot(x_values, ustar_values)
plt.xlabel('x')
plt.ylabel('ustar_t([x, 0, 0], t)')
plt.title('Plot of ustar_t([x, 0, 0], t) for x in range -5 to 5')
plt.grid(True)
plt.show()

#%%
# Sample 

n_samples = 1000
X = rejection_sample(n_samples, t, ustar)

#%%
from sbtm import losses, models
from importlib import reload
# reload(losses)

# Example usage
s = models.MLP(d)
weighting = jax.vmap(lambda x: D(x, X))(X)

target_score_values = jax.vmap(target_score)(X)

weighted_loss = losses.weighted_explicit_score_matching_loss(s, X, target_score_values, weighting)
print("Loss with D:", weighted_loss)

unweighted_loss = losses.explicit_score_matching_loss(s, X, target_score_values)
print("Loss with I:", unweighted_loss)

identity_weighting = jnp.array([jnp.eye(d) for _ in range(len(X))])
unweighted_loss = losses.weighted_explicit_score_matching_loss(s, X, target_score_values, identity_weighting)
print("Loss with I (weighted):", unweighted_loss)

# Compute the average trace of D[i]
average_trace = jnp.mean(jax.vmap(jnp.trace)(weighting))

# Divide the loss by the average trace
normalized_loss = d * weighted_loss / average_trace
print("Normalized Loss:", normalized_loss)

# %%

s1 = models.MLP(d)
s2 = models.MLP(d)

# Compute gradients of weighted and unweighted loss with respect to s1
_, grads_weighted_loss = nnx.value_and_grad(losses.weighted_explicit_score_matching_loss)(s1, X, target_score_values, weighting)
_, grads_unweighted_loss = nnx.value_and_grad(losses.explicit_score_matching_loss)(s1, X, target_score_values)

# Normalize gradients to unit length
normalize = lambda g: g / jnp.linalg.norm(g)
normalized_grads_weighted = jax.tree.map(normalize, grads_weighted_loss)
normalized_grads_unweighted = jax.tree.map(normalize, grads_unweighted_loss)

# Compute similarity between normalized gradients
similarity = jax.tree.map(lambda g1, g2: jnp.dot(g1.flatten(), g2.flatten()), normalized_grads_weighted, normalized_grads_unweighted)
average_similarity = jnp.mean(jnp.array([jnp.mean(v) for v in jax.tree.leaves(similarity)]))

print("Average similarity between normalized gradients:", average_similarity)

#%%
# Training parameters
learning_rate = 0.001
num_epochs = 1000
print_every_n = 10

s1 = models.MLP(d)

# Optimizer for s1
opt_1 = nnx.Optimizer(s1, optax.adamw(learning_rate, 0.9))

# Training loop for s1
weighted_losses_s1 = []
unweighted_losses_s1 = []

# Record initial losses for s1
weighted_loss_s1 = losses.weighted_explicit_score_matching_loss(s1, X, target_score_values, weighting)
unweighted_loss_s1 = losses.explicit_score_matching_loss(s1, X, target_score_values)

weighted_losses_s1.append(weighted_loss_s1)
unweighted_losses_s1.append(unweighted_loss_s1)

for epoch in tqdm(range(num_epochs), desc="Training s1"):
    # Compute losses and gradients for s1
    batch_size = 100
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        target_score_batch = target_score_values[i:i + batch_size]
        weighting_batch = weighting[i:i + batch_size]

        _, grads_s1 = nnx.value_and_grad(losses.weighted_explicit_score_matching_loss)(s1, X_batch, target_score_batch, weighting_batch)
        opt_1.update(grads_s1)
        
    # Record losses for s1
    weighted_loss_s1 = losses.weighted_explicit_score_matching_loss(s1, X, target_score_values, weighting)
    unweighted_loss_s1 = losses.explicit_score_matching_loss(s1, X, target_score_values)
    
    weighted_losses_s1.append(weighted_loss_s1)
    unweighted_losses_s1.append(unweighted_loss_s1)
    
    # Print losses for s1
    if epoch % print_every_n == 0:
        print(f"Epoch {epoch}: Weighted Loss s1 = {weighted_loss_s1:.4f}, Unweighted Loss s1 = {unweighted_loss_s1:.4f}")

s2 = models.MLP(d)

# Optimizer for s2
opt_2 = nnx.Optimizer(s2, optax.adamw(learning_rate, 0.9))

# Training loop for s2
weighted_losses_s2 = []
unweighted_losses_s2 = []

# Record initial losses for s2
weighted_loss_s2 = losses.weighted_explicit_score_matching_loss(s2, X, target_score_values, weighting)
unweighted_loss_s2 = losses.explicit_score_matching_loss(s2, X, target_score_values)

weighted_losses_s2.append(weighted_loss_s2)
unweighted_losses_s2.append(unweighted_loss_s2)

for epoch in tqdm(range(num_epochs), desc="Training s2"):
    # Compute losses and gradients for s2
    batch_size = 100
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        target_score_batch = target_score_values[i:i + batch_size]

        _, grads_s2 = nnx.value_and_grad(losses.explicit_score_matching_loss)(s2, X_batch, target_score_batch)
        opt_2.update(grads_s2)
        
    # Record losses for s2
    weighted_loss_s2 = losses.weighted_explicit_score_matching_loss(s2, X, target_score_values, weighting)
    unweighted_loss_s2 = losses.explicit_score_matching_loss(s2, X, target_score_values)
    
    weighted_losses_s2.append(weighted_loss_s2)
    unweighted_losses_s2.append(unweighted_loss_s2)
    
    # Print losses for s2
    if epoch % print_every_n == 0:
        print(f"Epoch {epoch}: Weighted Loss s2 = {weighted_loss_s2:.4f}, Unweighted Loss s2 = {unweighted_loss_s2:.4f}")

#%%
# Plot weighted losses for s1 and s2 on the same plot
plt.figure(figsize=(10, 5))
plt.plot(weighted_losses_s1, label='trained on weighted loss')
plt.plot(weighted_losses_s2, label='trained on unweighted loss')
plt.xlabel('Epoch')
plt.ylabel('Weighted Loss')
plt.title('Weighted Losses')
plt.legend()
plt.grid(True)
plt.show()

# Plot unweighted losses for s1 and s2 on the same plot
plt.figure(figsize=(10, 5))
plt.plot(unweighted_losses_s1, label='trained on weighted loss')
plt.plot(unweighted_losses_s2, label='trained on unweighted loss')
plt.xlabel('Epoch')
plt.ylabel('Unweighted Loss')
plt.title('Unweighted Losses')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Find largest and smallest eigenvalues for each element in weighting
all_eigenvalues = []

for w in tqdm(weighting):
    eigvals = jnp.linalg.eigvalsh(w)
    all_eigenvalues.append(eigvals)

# Convert to a numpy array for easier plotting
all_eigenvalues = jnp.array(all_eigenvalues)



# # Plot all sorted eigenvalues
# plt.figure(figsize=(10, 5))
# for i in range(sorted_eigenvalues.shape[1]):
#     plt.plot(sorted_eigenvalues[:, i], label=f'Eigenvalue {i+1}')

# plt.xlabel('Index')
# plt.ylabel('Eigenvalue')
# plt.title('Sorted Eigenvalues of Weighting Matrices')
# plt.legend()
# plt.grid(True)
# plt.show()
# Plot histograms of smallest, second smallest, and largest eigenvalues
smallest_eigenvalues = all_eigenvalues[:, 0]
second_smallest_eigenvalues = all_eigenvalues[:, 1]
largest_eigenvalues = all_eigenvalues[:, -1]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(smallest_eigenvalues, bins=30, alpha=0.75, color='blue')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Smallest Eigenvalues')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.hist(second_smallest_eigenvalues, bins=30, alpha=0.75, color='green')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Second Smallest Eigenvalues')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.hist(largest_eigenvalues, bins=30, alpha=0.75, color='red')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.title('Histogram of Largest Eigenvalues')
plt.grid(True)

plt.tight_layout()
plt.show()