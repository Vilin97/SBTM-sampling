# %%
import sys
# sys.path.append('/home/vilin/tempered-langevin')

import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
from typing import Any, Tuple
import functools
import jax
from flax.training import checkpoints
from tqdm import tqdm
import optax
# import flax.training.train_state as train_state
import matplotlib.pyplot as plt
# import einops

import os

# ROOT_FOLDER = "/home/vilin/SBTM-sampling/neurips"
ROOT_FOLDER = "/mmfs1/gscratch/krishna/psushko/SBTM-sampling/neurips"
IMAGE_FOLDER = os.path.join(ROOT_FOLDER, "images")

# %%
# Visualization
def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    result = (array.reshape(nrows, ncols, height, width, intensity)
            .swapaxes(1, 2)
            .reshape(height * nrows, width * ncols, intensity))
    return result


# %%
# Define the score model
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    embed_dim: int
    scale: float = 30.

    @nn.compact
    def __call__(self, x):
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim // 2, ))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    output_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)[:, None, None, :]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    Args:
        marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    marginal_prob_std: Any
    channels: Tuple[int] = (32, 64, 128, 256)
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x, t):
        # The swish activation function
        act = nn.swish
        # Obtain the Gaussian random feature embedding for t
        embed = act(nn.Dense(self.embed_dim)(
            GaussianFourierProjection(embed_dim=self.embed_dim)(t)))

        # Encoding path
        h1 = nn.Conv(self.channels[0], (3, 3), (1, 1), padding='VALID',
                     use_bias=False)(x)
        # # Incorporate information from t
        h1 += Dense(self.channels[0])(embed)
        # # Group normalization
        h1 = nn.GroupNorm(4)(h1)
        h1 = act(h1)
        h2 = nn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID',
                     use_bias=False)(h1)
        h2 += Dense(self.channels[1])(embed)
        h2 = nn.GroupNorm()(h2)
        h2 = act(h2)
        h3 = nn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID',
                     use_bias=False)(h2)
        h3 += Dense(self.channels[2])(embed)
        h3 = nn.GroupNorm()(h3)
        h3 = act(h3)
        h4 = nn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID',
                     use_bias=False)(h3)
        h4 += Dense(self.channels[3])(embed)
        h4 = nn.GroupNorm()(h4)
        h4 = act(h4)

        # Decoding path
        h = nn.Conv(self.channels[2], (3, 3), (1, 1), padding=((2, 2), (2, 2)),
                    input_dilation=(2, 2), use_bias=False)(h4)
        # # Skip connection from the encoding path
        h += Dense(self.channels[2])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(self.channels[1], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                    input_dilation=(2, 2), use_bias=False)(
                        jnp.concatenate([h, h3], axis=-1)
        )
        h += Dense(self.channels[1])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(self.channels[0], (3, 3), (1, 1), padding=((2, 3), (2, 3)),
                    input_dilation=(2, 2), use_bias=False)(
                        jnp.concatenate([h, h2], axis=-1)
        )
        h += Dense(self.channels[0])(embed)
        h = nn.GroupNorm()(h)
        h = act(h)
        h = nn.Conv(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
            jnp.concatenate([h, h1], axis=-1)
        )

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
    t: A vector of time steps.
    sigma: The sigma in our SDE.

    Returns:
    The standard deviation.
    """
    return jnp.sqrt((sigma**(2 * t) - 1.) / 2. / jnp.log(sigma))


sigma = 25.
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

# %%
# Load the score model
rng = jax.random.PRNGKey(0)
fake_input = jnp.ones((2, 28, 28, 1))
fake_time = jnp.ones(2)
score_model = ScoreNet(marginal_prob_std_fn)
params = score_model.init({'params': rng}, fake_input, fake_time)
optimizer = optax.adam(learning_rate=0.001)

train_state = checkpoints.restore_checkpoint(ROOT_FOLDER + "/checkpoint_750", target=None, step=None)

params = train_state["params"]

def target_score(x):
    """Computes the score of the target distribution.
    Args:
      x: A batch of samples.
         shape = (batch_size, 28 * 28 * 1)
    """
    x = x.reshape((x.shape[0], 28, 28, 1))
    t = jnp.ones(x.shape[0],) * 0.001
    score_val = score_model.apply(params, x, t)
    return score_val

# %%
# Optimization
sample_batch_size = 128
# samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.01
# samples = jnp.zeros((sample_batch_size, 28, 28, 1))

step_size = 0.001
num_steps = 81

for i in tqdm(range(num_steps)):
    samples = samples + step_size * target_score(samples)
    if i % 20 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()

plt.imshow(gallery(samples, 16))

#%%
# Langevin
rng = jax.random.PRNGKey(1)
sample_batch_size = 128
samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
# samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.01
# samples = jnp.zeros((sample_batch_size, 28, 28, 1))

step_size = 0.001
num_steps = 81

for i in tqdm(range(num_steps)):
    step_rng, rng = jax.random.split(rng)
    samples = samples + step_size * target_score(samples)
    samples += jnp.sqrt(2 * step_size) * jax.random.normal(step_rng, samples.shape)
    if i % 20 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()

plt.imshow(gallery(samples, 16))

#%%
# Plan for SBTM:
# 1. Take the initial model to be of the same architecture as the score model.
# 2. Train it on the score of the initial distribution -- either a Gaussian or a uniform distribution.
# 3. Change stepping of Langevin to use the new model instead of the noise.
# 4. Also, train the model on the implicit score match ing loss at every step.
# 5. PROFIT.

#%%
class ScoreNetStatic(nn.Module):
    """U-Net–style score model without time conditioning."""
    channels: Tuple[int] = (32, 64, 128, 256)

    @nn.compact
    def __call__(self, x):
        act = nn.swish

        # Encoder
        h1 = act(nn.GroupNorm(4)(nn.Conv(self.channels[0], (3, 3), padding='VALID', use_bias=False)(x)))
        h2 = act(nn.GroupNorm()(nn.Conv(self.channels[1], (3, 3), (2, 2), padding='VALID', use_bias=False)(h1)))
        h3 = act(nn.GroupNorm()(nn.Conv(self.channels[2], (3, 3), (2, 2), padding='VALID', use_bias=False)(h2)))
        h4 = act(nn.GroupNorm()(nn.Conv(self.channels[3], (3, 3), (2, 2), padding='VALID', use_bias=False)(h3)))

        # Decoder
        h = act(nn.GroupNorm()(nn.Conv(self.channels[2], (3, 3), padding=((2, 2), (2, 2)),
                                      input_dilation=(2, 2), use_bias=False)(h4)))
        h = act(nn.GroupNorm()(nn.Conv(self.channels[1], (3, 3), padding=((2, 3), (2, 3)),
                                      input_dilation=(2, 2), use_bias=False)(
            jnp.concatenate([h, h3], axis=-1))))
        h = act(nn.GroupNorm()(nn.Conv(self.channels[0], (3, 3), padding=((2, 3), (2, 3)),
                                      input_dilation=(2, 2), use_bias=False)(
            jnp.concatenate([h, h2], axis=-1))))
        out = nn.Conv(1, (3, 3), padding=((2, 2), (2, 2)))(
            jnp.concatenate([h, h1], axis=-1))

        return out

#%%
rng = jax.random.PRNGKey(1)
sample_batch_size = 128
samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))

#%%
# Initialize ScoreNetStatic and evaluate on samples
s = ScoreNetStatic()
static_params = s.init(rng, samples)
static_out = s.apply(static_params, samples)
print("ScoreNetStatic output shape:", static_out.shape)

# %%
import jax, jax.numpy as jnp
from functools import partial
import time

def get_div_fn(fn):
    """Return div_fn(x, t, eps) that produces a *scalar* divergence estimate."""
    def div_fn(x, t, eps):
        g = lambda d: jnp.sum(fn(d, t) * eps)          # scalar ↑
        return jnp.vdot(jax.grad(g)(x), eps)           # scalar divergence
    return div_fn

# ---------------------------------------------------------------------
# 1.  Build a divergence function tied to *apply_fn(params, x)*.
#    get_div_fn expects     fn(x, t)         so we curry params into it
#    and ignore the dummy   t.
# ---------------------------------------------------------------------
def make_div_fn(params, apply_fn):
    score_fn = lambda x, _t: apply_fn(params, x)        # _t is unused
    return get_div_fn(score_fn)                         # returns div_fn(x, t, eps)


# ---------------------------------------------------------------------
# 2.  Jitted ISM loss using that div_fn and n_hutch Hutchinson probes
# ---------------------------------------------------------------------
def ism_loss(params, apply_fn, particles, rng, *, n_hutch=4):
    B   = particles.shape[0]
    div = make_div_fn(params, apply_fn)                 # closes over params

    # keys: (B, n_hutch, 2)  for per-sample, per-probe randomness
    keys = jax.random.split(rng, B * n_hutch).reshape(B, n_hutch, 2)

    def per_sample(x, ks):
        # --- Hutchinson divergence (avg over n_hutch) ----------------
        def one_probe(k):
            eps = jax.random.rademacher(k, x.shape, dtype=x.dtype)
            return div(x, None, eps)                    # t=None
        div_est = jax.vmap(one_probe)(ks).mean()

        s = apply_fn(params, x)                         # score
        return 0.5 * jnp.sum(s**2) + div_est           # per-sample loss

    loss = jax.vmap(per_sample)(particles, keys).mean()
    return loss

# ---------------------------------------------------------------------
# 3.  JIT-compile once, use inside your training loop
# ---------------------------------------------------------------------
fast_ism_loss = jax.jit(partial(ism_loss, n_hutch=1), static_argnames='apply_fn')

rng        = jax.random.PRNGKey(0)
loss_value = fast_ism_loss(static_params, s.apply, samples, rng)
print("loss:", loss_value)

# If you need gradients:
loss_val, grads = jax.value_and_grad(fast_ism_loss)(
        static_params, s.apply, samples, rng)

#%%

start1 = time.time()
loss_value1 = fast_ism_loss(static_params, s.apply, samples, jax.random.PRNGKey(0)).block_until_ready()
end1 = time.time()
print("Time taken for first call:", end1 - start1)

start2 = time.time()
loss_val2, grads2 = jax.value_and_grad(fast_ism_loss)(
    static_params, s.apply, samples, jax.random.PRNGKey(0))
end2 = time.time()
print("Time taken for value_and_grad call:", end2 - start2)

# %%
# ── quick sanity checks for ISM loss ────────────────────────────
import jax, jax.numpy as jnp

# (1)  Uniform data, score ≡ 0  →  loss should be 0
loss_zero = fast_ism_loss({}, lambda _p, x: jnp.zeros_like(x),
                          samples, jax.random.PRNGKey(13))
print("uniform + zero‑score  loss:", loss_zero)          # ≈ 0

# (2)  N(0,1) data, exact Gaussian score  s(x) = −x
gauss_samples = jax.random.normal(jax.random.PRNGKey(17),
                                  (sample_batch_size, 28, 28, 1))
d = 28 * 28                                              # dimensionality
loss_gauss = fast_ism_loss({}, lambda _p, x: -x,
                           gauss_samples, jax.random.PRNGKey(19))
print("gaussian + exact‑score loss:", loss_gauss)        # ≈ -0.5 * d
print("expected:", -0.5 * d)


# %%

d = 28 * 28                          # dimensionality of one sample
B = sample_batch_size

def gaussian_test(sigma2=1.0, shift=0.0, seed=0):
    key = jax.random.PRNGKey(seed)
    x   = shift + jnp.sqrt(sigma2) * jax.random.normal(key, (B, 28, 28, 1))
    score = lambda _p, z: -(z - shift) / sigma2
    loss  = fast_ism_loss({}, score, x, key)
    target = -0.5 * d / sigma2          # analytic value (independent of shift)
    print(f"σ²={sigma2:4.1f}, shift={shift:4.1f}  →  loss={loss:.3f},  expected={target:.3f}")

# isotropic Gaussians with different variances
gaussian_test(1.0)            # should be ≈ -392.0
gaussian_test(4.0)            # should be ≈ -98.0
gaussian_test(0.5)            # should be ≈ -784.0

# %%
loss_zero = fast_ism_loss({}, lambda _p, x: jnp.zeros_like(x),
                          samples, jax.random.PRNGKey(13))
print("uniform + zero‑score  loss:", loss_zero)          # ≈ 0

# (2)  N(0,1) data, exact Gaussian score  s(x) = −x
gauss_samples = jax.random.normal(jax.random.PRNGKey(17),
                                  (sample_batch_size, 28, 28, 1))
d = 28 * 28                                              # dimensionality
loss_gauss = fast_ism_loss({}, lambda _p, x: -x,
                           gauss_samples, jax.random.PRNGKey(19))
print("gaussian + exact‑score loss:", loss_gauss)        # ≈ -0.5 * d
print("expected:", -0.5 * d)



# %%
import jax, jax.numpy as jnp, optax, matplotlib.pyplot as plt
from flax import serialization
from tqdm import trange
import jax, jax.numpy as jnp, optax, matplotlib.pyplot as plt
from flax import serialization
from tqdm import trange

# --- config ----------------------------------------------------
rng          = jax.random.PRNGKey(0)
batch_size   = 512
steps        = 5_000          # adjust if you want longer training
lr           = 1e-3
shape        = (batch_size, 28, 28, 1)

# --- model, optimiser, loss -----------------------------------
model   = ScoreNetStatic()
params  = model.init(rng, jax.random.uniform(rng, shape))
opt     = optax.adam(lr)
opt_st  = opt.init(params)

@jax.jit
def train_step(p, ostate, key):
    x = jax.random.uniform(key, shape)
    loss, grads = jax.value_and_grad(lambda q: fast_ism_loss(q, model.apply, x, key))(p)
    updates, ostate = opt.update(grads, ostate, p)
    p = optax.apply_updates(p, updates)
    return p, ostate, loss

# --- training --------------------------------------------------
loss_history = []
for _ in trange(steps, desc="Training"):
    rng, sub = jax.random.split(rng)
    params, opt_st, loss = train_step(params, opt_st, sub)
    loss_history.append(loss)

# --- save ------------------------------------------------------
with open("score_uniform_params.msgpack", "wb") as f:
    f.write(serialization.to_bytes(params))

# --- visualize -------------------------------------------------
plt.figure()
plt.plot(loss_history)
plt.xlabel("step")
plt.ylabel("ISM loss")
plt.title("Training ScoreNetStatic on Uniform Data")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- config ----------------------------------------------------
rng          = jax.random.PRNGKey(0)
batch_size   = 512
steps        = 5_000          # adjust if you want longer training
lr           = 1e-3
shape        = (batch_size, 28, 28, 1)

# --- model, optimiser, loss -----------------------------------
model   = ScoreNetStatic()
params  = model.init(rng, jax.random.uniform(rng, shape))
opt     = optax.adam(lr)
opt_st  = opt.init(params)

@jax.jit
def train_step(p, ostate, key):
    x = jax.random.uniform(key, shape)
    loss, grads = jax.value_and_grad(lambda q: fast_ism_loss(q, model.apply, x, key))(p)
    updates, ostate = opt.update(grads, ostate, p)
    p = optax.apply_updates(p, updates)
    return p, ostate, loss

# --- training --------------------------------------------------
loss_history = []
for _ in trange(steps, desc="Training"):
    rng, sub = jax.random.split(rng)
    params, opt_st, loss = train_step(params, opt_st, sub)
    loss_history.append(loss)

# --- save ------------------------------------------------------
with open("score_uniform_params.msgpack", "wb") as f:
    f.write(serialization.to_bytes(params))

# --- visualize -------------------------------------------------
plt.figure()
plt.plot(loss_history)
plt.xlabel("step")
plt.ylabel("ISM loss")
plt.title("Training ScoreNetStatic on Uniform Data")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

'''

Plan

load prior model

get examplmes, 

play with hyperparams



'''