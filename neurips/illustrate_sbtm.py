# %%
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
from sbtm import distribution 

from sbtm import models, losses, sampler
import optax
from flax import nnx

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#%%
s = 0.2
f_dist = distribution.Gaussian(jnp.zeros(2), jnp.eye(2)*s)
f_pdf = f_dist.density
f_score = f_dist.score

# g_dist = distribution.Gaussian(jnp.ones(2)*1.2, jnp.eye(2)*0.7)
g_dist = distribution.GaussianMixture([jnp.array([-0.2, 1.2]), jnp.array([1.8, -0.3])], [jnp.eye(2)*0.5, jnp.eye(2)*0.5], [0.5, 0.5])
g_pdf = g_dist.density
g_score = g_dist.score

x = jrandom.normal(jrandom.PRNGKey(0), (100, 2)) * s
# Create a grid for heatmaps
xx, yy = jnp.meshgrid(jnp.linspace(-1, 2, 200), jnp.linspace(-0.8, 2, 200))
grid = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

# Vectorized evaluation of pdfs on the grid
f_vals = jnp.reshape(f_pdf(grid), xx.shape)
g_vals = jnp.reshape(g_pdf(grid), xx.shape)
f_score_vals = f_score(x)
g_score_vals = g_score(x)

plt.figure(figsize=(4, 4))
plt.contourf(xx, yy, f_vals, levels=30, alpha=0.5, cmap='Greens')
plt.contourf(xx, yy, g_vals, levels=30, alpha=0.5, cmap='Reds')
plt.scatter(x[:, 0], x[:, 1], c='k', s=10)

# Find the rightmost point p (largest x coordinate)
idx = jnp.argmax(x[:, 0])
step_size = 1/6.5
bm = jnp.sqrt(2) * jrandom.normal(jrandom.PRNGKey(15), x.shape)

# Draw arrows for 5 more randomly chosen points
key = jrandom.PRNGKey(5)
rand_indices = jrandom.choice(key, x.shape[0], shape=(5,), replace=False)
rand_indices = jnp.concatenate((rand_indices, jnp.array([idx])))  # Ensure the rightmost point is included

for i, idx in enumerate(rand_indices):
    p_rand = x[idx]
    g_score_val = g_score_vals[idx] * step_size
    f_score_val = -f_score_vals[idx] * step_size
    bm_val = bm[idx] * jnp.sqrt(step_size)
    g_label = r'$\nabla \log \pi$' if i == 0 else None
    f_label = r'$-\nabla \log f_t$' if i == 0 else None
    bm_label = r'$\sqrt{2}\mathrm{d} B_t$' if i == 0 else None
    plt.arrow(p_rand[0], p_rand[1], g_score_val[0], g_score_val[1], color='red', width=0.01, head_width=0.07, length_includes_head=True, label=g_label)
    plt.arrow(p_rand[0], p_rand[1], f_score_val[0], f_score_val[1], color='green', width=0.01, head_width=0.07, length_includes_head=True, label=f_label)
    plt.arrow(p_rand[0], p_rand[1], bm_val[0], bm_val[1], color='blue', width=0.01, head_width=0.07, length_includes_head=True, label=bm_label)
    plt.scatter([p_rand[0]], [p_rand[1]], c='yellow', s=60, edgecolors='k', zorder=5)

plt.legend(loc='upper right')
plt.axis('off')
plt.show()

#%%
dt = 0.2
x_new_f = x + dt * (g_score_vals - f_score_vals)
x_new_bm = x + jnp.sqrt(dt) * bm

plt.figure(figsize=(4, 4))
plt.contourf(xx, yy, f_vals, levels=30, alpha=0.5, cmap='Greens')
plt.contourf(xx, yy, g_vals, levels=30, alpha=0.5, cmap='Reds')
plt.scatter(x[:, 0], x[:, 1], c='gray', s=10, alpha=0.4, label=r'$X_t^i$')
plt.scatter(x_new_f[:, 0], x_new_f[:, 1], c='g', s=10, label='SBTM')
# Only plot points inside the boundary of the image
mask = (
    (x_new_bm[:, 0] >= -1) & (x_new_bm[:, 0] <= 2) &
    (x_new_bm[:, 1] >= -0.8) & (x_new_bm[:, 1] <= 2)
)
plt.scatter(x_new_bm[mask, 0], x_new_bm[mask, 1], c='b', s=10, label='SDE')

# Draw arrows from x to x_new_f (SBTM update)
for i, idx in enumerate(rand_indices):
    start = x[idx]
    end = x_new_f[idx]
    delta = end - start
    label = 'SBTM update' if i == 0 else None
    plt.arrow(start[0], start[1], delta[0], delta[1], color='green', width=0.01, head_width=0.07, length_includes_head=True, label=label)
    plt.scatter([end[0]], [end[1]], c='yellow', s=60, edgecolors='k', zorder=5)

# Draw arrows from x to x_new_bm (SDE update)
for i, idx in enumerate(rand_indices):
    start = x[idx]
    end = x_new_bm[idx]
    delta = end - start
    label = 'SDE update' if i == 0 else None
    plt.arrow(start[0], start[1], delta[0], delta[1], color='blue', width=0.01, head_width=0.07, length_includes_head=True, label=label, alpha=0.7)
    plt.scatter([end[0]], [end[1]], c='orange', s=60, edgecolors='k', zorder=5)
    
    

plt.legend(loc='upper right')
plt.axis('off')
plt.show()

# %%
from sbtm import models, losses, sampler
import optax
from flax import nnx

x_aug = jnp.concatenate([x, jrandom.normal(jrandom.PRNGKey(10), (10000, 2)) * s], axis=0)

score_model = models.MLP(2)
# loss = losses.implicit_score_matching_loss
loss = lambda model, x: losses.explicit_score_matching_loss(model, x, f_score(x))
optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))

#%%
trainsteps = 500
loss = losses.implicit_score_matching_loss
for i in range(trainsteps):
    sampler.opt_step(score_model, optimizer, loss, x_aug)

#%%

plt.figure(figsize=(4, 4))
plt.contourf(xx, yy, f_vals, levels=30, alpha=0.5, cmap='Greens')
plt.contourf(xx, yy, g_vals, levels=30, alpha=0.5, cmap='Reds')
plt.scatter(x_aug[:, 0], x_aug[:, 1], c='k', s=10)

for i, idx in enumerate(rand_indices):
    p_rand = x_aug[idx]
    s_dir_rand = -score_model(p_rand.reshape(1,2)).flatten() * step_size
    f_score_val = -f_score_vals[idx] * step_size
    s_label = '-learned score' if i == 0 else None
    f_label = '-true score' if i == 0 else None
    
    plt.arrow(p_rand[0], p_rand[1], s_dir_rand[0], s_dir_rand[1], color='green', width=0.01, head_width=0.07, length_includes_head=True, label=s_label)
    plt.arrow(p_rand[0], p_rand[1], f_score_val[0], f_score_val[1], color='red', width=0.01, head_width=0.07, length_includes_head=True, label=f_label)
    plt.scatter([p_rand[0]], [p_rand[1]], c='yellow', s=60, edgecolors='k', zorder=5)

plt.legend(loc='upper right')
plt.axis('off')
plt.title(f'Learned score after {trainsteps} implicit steps')
plt.show()
# %%
