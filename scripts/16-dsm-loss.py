# %%
import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import trange

from sbtm import distribution 
from sbtm import models, losses, sampler
import optax
from flax import nnx
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# %%
"Sample Gaussian Mixture"
f_dist = distribution.GaussianMixture([jnp.array([-2.5,0]), jnp.array([2.5,0])], [jnp.eye(2), jnp.eye(2)], [0.5, 0.5])
f_pdf = f_dist.density
f_score = f_dist.score

key = jrandom.PRNGKey(0)
x = f_dist.sample(key, 2**12)
# Create a grid for heatmaps
xx, yy = jnp.meshgrid(jnp.linspace(-6, 6, 200), jnp.linspace(-3, 3, 200))
grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

# Vectorized evaluation of pdfs on the grid
f_vals = jnp.reshape(f_pdf(grid), xx.shape)
f_score_vals = f_score(x)
key = jrandom.PRNGKey(7)
rand_indices = jrandom.choice(key, x.shape[0], shape=(20,), replace=False)

#%%
# DDPM-style noise prediction loss

class CondMLP(nnx.Module):
    """MLP that conditions on σ (scalar or (B,) or (B,1))."""
    def __init__(self, d_x, hidden_units=[128, 128],
                 activation=nnx.soft_sign, seed=0):
        rngs = nnx.Rngs(seed)
        layers, inp = [], d_x + 1           # +1 for σ
        for h in hidden_units:
            layers.append(nnx.Linear(inp, h, rngs=rngs))
            inp = h
        self.hidden, self.out, self.act = layers, nnx.Linear(inp, d_x, rngs=rngs), activation

    def __call__(self, x, sigma):           # x: (B,d), sigma: scalar or array
        sigma = jnp.asarray(sigma, dtype=x.dtype)
        if sigma.ndim == 0:                 # make (B,1)
            sigma = jnp.broadcast_to(sigma, (x.shape[0], 1))
        elif sigma.ndim == 1:               # (B,) -> (B,1)
            sigma = sigma.reshape(-1, 1)

        h = jnp.concatenate([x, sigma], axis=-1)
        for layer in self.hidden:
            h = self.act(layer(h))
        return self.out(h)

@nnx.jit
def dsm_loss(model, x, rng, σ_min=0.1, σ_max=2.0):
    k_σ, k_ε = jrandom.split(rng)
    σ = jrandom.uniform(k_σ, shape=(x.shape[0],1), minval=σ_min, maxval=σ_max)
    ε        = jrandom.normal(k_ε, x.shape)
    x_noisy  = x + σ * ε
    pred     = model(x_noisy, σ)
    return 0.5 * jnp.mean(jnp.sum((pred + ε/σ)**2, -1) * σ**2)

@nnx.jit
def explicit_loss(model, x, f_score_x, σ=0.5):
    pred = model(x, σ)
    return jnp.sum((pred - f_score_x) ** 2) / x.shape[0]

@nnx.jit(static_argnames=('loss'))
def train_step(model, optimizer, loss, x, rng):
    """Single training step. Returns loss, rng, grad norm."""
    rng, step_rng = jrandom.split(rng)
    loss_val, grads = nnx.value_and_grad(loss)(model, x, step_rng)
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
    optimizer.update(grads)
    return loss_val, rng, grad_norm

# Train
loss, loss_name = dsm_loss, 'DSM Loss'
score_model = CondMLP(d_x=2, hidden_units=[128, 128, 128])
# optimizer = nnx.Optimizer(score_model, optax.adam(2e-3, 0.99))
optimizer = nnx.Optimizer(score_model, optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, b1=0.9, b2=0.99),))

sigmas = [0.1, 0.5]
explicit_loss_vals = {sigma: [] for sigma in sigmas}
loss_vals = []
fisher_vals = []
grad_norms = []

trainsteps = 5000
rng = jrandom.PRNGKey(0)
f_score_x = f_score(x)  # true score

for i in trange(trainsteps):
    loss_val, rng, grad_norm = train_step(score_model, optimizer, loss, x, rng)
    loss_vals.append(loss_val)
    for sigma in sigmas:
        explicit_loss_vals[sigma].append(explicit_loss(score_model, x, f_score_x, sigma))
    grad_norms.append(grad_norm)

# Compute EMA of loss_vals with smoothing=0.95
def ema(values, smoothing=0.95):
    ema_vals = []
    prev = values[0]
    for v in values:
        prev = smoothing * prev + (1 - smoothing) * v
        ema_vals.append(prev)
    return ema_vals

#%%
plt.figure(figsize=(8, 2.5))
plt.plot(loss_vals, label=f'{loss_name}')
plt.plot(ema(loss_vals, 0.95), label=f'EMA(0.95) {loss_name}')
plt.legend()
plt.show()

plt.figure(figsize=(8, 2.5))
plt.plot(grad_norms, label='Gradient Norm')
plt.plot(ema(grad_norms, 0.95), label='EMA(0.95) Gradient Norm')
plt.legend()
plt.show()

plt.figure(figsize=(8, 2.5))
for sigma in sigmas:
    vals = jnp.array(explicit_loss_vals[sigma])
    plt.plot(vals, label=f'MSE σ={sigma}, avg={jnp.mean(vals):.2f}')
    ema_explicit = ema(vals, 0.95)
    plt.plot(ema_explicit, label=f'EMA(0.95) σ={sigma}')
    min_idx = jnp.argmin(jnp.array(ema_explicit))
    plt.scatter(min_idx, ema_explicit[min_idx], color='red', s=60, zorder=10, label=f'σ={sigma} min={ema_explicit[min_idx]:.2f}')
plt.axhline(0, color='black', linestyle='dotted', linewidth=1)
plt.xlabel('Iteration')
plt.title(f'Training on {loss_name} (Explicit Loss for each σ)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.contourf(xx, yy, f_vals, levels=30, alpha=0.5, cmap='Greens')
plt.scatter(x[:, 0], x[:, 1], c='k', s=10)

for i, idx in enumerate(rand_indices):
    p_rand = x[idx]
    s_dir_rand = -score_model(p_rand.reshape(1,2), 0.1).flatten()
    f_score_val = -f_score_vals[idx]
    s_label = '-learned score' if i == 0 else None
    f_label = '-true score' if i == 0 else None
    
    plt.arrow(p_rand[0], p_rand[1], f_score_val[0], f_score_val[1], color='red', width=0.01, head_width=0.07, length_includes_head=True, label=f_label)
    plt.arrow(p_rand[0], p_rand[1], s_dir_rand[0], s_dir_rand[1], color='green', width=0.01, head_width=0.07, length_includes_head=True, label=s_label)
    plt.scatter([p_rand[0]], [p_rand[1]], c='yellow', s=60, edgecolors='k', zorder=5)

plt.legend(loc='upper right')
plt.axis('off')
plt.title(f'Learned score after {trainsteps} steps on {loss_name}')
plt.show()

# %%
# TODO: make plot of explicit loss vs bandwidth
"Experiment: KDE score approximation vs true score"

import jax.scipy.stats as jsp_stats

# --- Bandwidth function ---
def bandwidth(x, all_x, k=5):
    """
    Returns the bandwidth for point x given all data points all_x.
    Uses the average distance to the k nearest neighbors (excluding itself).
    x: shape (d,)
    all_x: shape (N, d)
    """
    # Compute distances from x to all points in all_x
    dists = np.linalg.norm(all_x - x, axis=1)
    # Exclude zero distance (itself)
    dists = dists[dists > 0]
    # Take k nearest
    if len(dists) < k:
        k = len(dists)
    nearest = np.partition(dists, k-1)[:k]
    return np.mean(nearest)

# Use the same data as before
kde_sigmas = [0.5]
explicit_loss_vals_kde = {}

for sigma in kde_sigmas:
    dataset = np.array(x).T  # shape (2, N)
    # Compute per-point bandwidths
    all_x = np.array(x)
    bandwidths = np.array([bandwidth(pt, all_x, k=45) for pt in all_x])
    # For now, use the mean bandwidth for the whole dataset (since gaussian_kde expects a scalar/callable)
    mean_bw = np.mean(bandwidths)
    bw = lambda kde_self: mean_bw
    kde = jsp_stats.gaussian_kde(dataset, bw_method=bw)
    # KDE density and score using JAX
    def kde_density(x_):
        # x_: (N, 2)
        return kde.pdf(jnp.asarray(x_).T)
    def kde_score(x_):
        # x_: (N, 2)
        def log_kde_density(x_single):
            # x_single: (2,)
            # kde.logpdf returns shape (1,), so extract scalar
            return kde.logpdf(x_single).squeeze()
        grad_fn = jax.vmap(jax.grad(log_kde_density))
        return np.array(grad_fn(jnp.asarray(x_)))
    # Compute explicit loss (MSE between KDE score and true score)
    kde_score_vals = kde_score(np.array(x))
    explicit_loss = np.mean(np.sum((kde_score_vals - np.array(f_score_vals))**2, axis=1))
    explicit_loss_vals_kde[sigma] = explicit_loss

    # Scatterplot: compare KDE score and true score
    plt.figure(figsize=(8, 4))
    plt.contourf(xx, yy, f_vals, levels=30, alpha=0.5, cmap='Greens')
    plt.scatter(x[:, 0], x[:, 1], c='k', s=10)
    for i, idx in enumerate(rand_indices):
        p_rand = x[idx]
        kde_score_val = -kde_score_vals[idx]
        f_score_val = -f_score_vals[idx]
        kde_label = '-KDE score' if i == 0 else None
        f_label = '-true score' if i == 0 else None
        plt.arrow(p_rand[0], p_rand[1], f_score_val[0], f_score_val[1], color='red', width=0.01, head_width=0.07, length_includes_head=True, label=f_label)
        plt.arrow(p_rand[0], p_rand[1], kde_score_val[0], kde_score_val[1], color='green', width=0.01, head_width=0.07, length_includes_head=True, label=kde_label)
        plt.scatter([p_rand[0]], [p_rand[1]], c='yellow', s=60, edgecolors='k', zorder=5)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.title(f'KDE score (σ={sigma}), Explicit Loss={explicit_loss:.3f}')
    plt.show()

# Print explicit loss for each bandwidth
for sigma in kde_sigmas:
    print(f"KDE bandwidth σ={sigma}: Explicit Loss={explicit_loss_vals_kde[sigma]:.4f}")

# %%
# --- Bandwidth function ---
def compute_bandwidths(all_x, k=10):
    """
    Compute per-point bandwidths using average distance to k nearest neighbors.
    all_x: (N, d)
    Returns: (N,) array of bandwidths
    """
    # Compute full pairwise distances
    diffs = all_x[:, None, :] - all_x[None, :, :]    # (N, N, d)
    dists = jnp.linalg.norm(diffs, axis=-1)          # (N, N)
    # Exclude self (set diagonal to large value)
    dists = dists + jnp.eye(all_x.shape[0]) * 1e6
    # Find k smallest distances per point (axis 1)
    nearest = jnp.sort(dists, axis=1)[:, :k]
    return jnp.mean(nearest, axis=1)                 # (N,)

# --- KDE density and score functions ---
def kde_density(x_query, x_train, sigmas):
    """
    Evaluate adaptive KDE at x_query using per-point sigmas.
    x_query: (M, d)
    x_train: (N, d)
    sigmas: (N,) bandwidths per training point
    Returns: (M,) KDE values
    """
    def eval_single(xq):
        diffs = xq - x_train            # (N, d)
        norms = jnp.sum(diffs ** 2, axis=1)  # (N,)
        kernels = jnp.exp(-norms / (2 * sigmas**2)) / ((2 * jnp.pi)**(xq.shape[0]/2) * sigmas**xq.shape[0])
        return jnp.mean(kernels)
    return jax.vmap(eval_single)(x_query)

def kde_score(x_query, x_train, sigmas):
    """
    Score = ∇ log p(x). Gradient of log-density estimate.
    x_query: (M, d)
    Returns: (M, d)
    """
    def log_density(xq):
        return jnp.log(kde_density(xq[None, :], x_train, sigmas) + 1e-12)[0]
    return jax.vmap(jax.grad(log_density))(x_query)

# --- Main experiment ---
"Experiment: Adaptive KDE score approximation vs true score"
explicit_loss_vals_kde = {}

all_x = jnp.array(x)
bandwidths = compute_bandwidths(all_x, k=10)
kde_score_vals = kde_score(all_x, all_x, bandwidths)
explicit_loss = jnp.mean(jnp.sum((kde_score_vals - jnp.array(f_score_vals))**2, axis=1))
explicit_loss_vals_kde["adaptive"] = float(explicit_loss)

# Scatterplot: compare KDE score and true score
plt.figure(figsize=(8, 4))
plt.contourf(xx, yy, f_vals, levels=30, alpha=0.5, cmap='Greens')
plt.scatter(x[:, 0], x[:, 1], c='k', s=10)
for i, idx in enumerate(rand_indices):
    p_rand = x[idx]
    kde_score_val = -kde_score_vals[idx]
    f_score_val = -f_score_vals[idx]
    kde_label = '-KDE score' if i == 0 else None
    f_label = '-true score' if i == 0 else None
    plt.arrow(p_rand[0], p_rand[1], f_score_val[0], f_score_val[1], color='red', width=0.01, head_width=0.07, length_includes_head=True, label=f_label)
    plt.arrow(p_rand[0], p_rand[1], kde_score_val[0], kde_score_val[1], color='green', width=0.01, head_width=0.07, length_includes_head=True, label=kde_label)
    plt.scatter([p_rand[0]], [p_rand[1]], c='yellow', s=60, edgecolors='k', zorder=5)
plt.legend(loc='upper right')
plt.axis('off')
plt.title(f'Adaptive KDE score, Explicit Loss={explicit_loss:.3f}')
plt.show()

# Print explicit loss
print(f"Adaptive KDE: Explicit Loss={explicit_loss:.4f}")