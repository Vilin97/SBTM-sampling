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
import einops

import os

ROOT_FOLDER = "/home/vilin/SBTM-sampling/neurips"
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
rng = jax.random.PRNGKey(1)
sample_batch_size = 128
# samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.01
# samples = jnp.zeros((sample_batch_size, 28, 28, 1))

step_size = 0.001
num_steps = 41

for i in tqdm(range(num_steps)):
    samples = samples + step_size * target_score(samples)
    if i % 10 == 0:
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
num_steps = 41

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

# %%
import jax, jax.numpy as jnp
from functools import partial

def get_div_fn(fn):
    def div_fn(x, eps):
        g = lambda y: jnp.vdot(fn(y), eps)        # scalar
        grad = jax.grad(g)(x)                     # Jᵀ eps
        return jnp.vdot(grad, eps)                # epsᵀJeps
    return div_fn

def make_fast_ism_loss(n_hutch: int = 4):
    @partial(jax.jit, static_argnums=(1,))        # 1 = apply_fn  (static)
    def _loss(params, apply_fn, particles, rng):
        B     = particles.shape[0]
        keys  = jax.random.split(rng, B * n_hutch).reshape(B, n_hutch, 2)
        div_f = get_div_fn(lambda y: apply_fn(params, y))

        def per_sample(x, ks):
            def one_probe(k):
                eps = jax.random.rademacher(k, x.shape, dtype=x.dtype)
                return div_f(x, eps)
            div_est = jax.vmap(one_probe)(ks).mean(0)
            s       = apply_fn(params, x)
            return 0.5 * jnp.sum(s**2) + div_est

        return jax.vmap(per_sample)(particles, keys).mean()

    return _loss

fast_ism_loss = make_fast_ism_loss(n_hutch=20)

#%%
def run_fast_ism_gd(
    params,
    apply_fn,
    samples,
    rng,
    optimizer,
    opt_state,
    fast_ism_loss,
    epochs=10,
    batch_size=32,
    verbose=True
):
    N = samples.shape[0]
    losses = []

    for epoch in range(epochs):
        # Shuffle samples
        rng, shuffle_rng, batch_rng = jax.random.split(rng, 3)
        idx = jax.random.permutation(shuffle_rng, N)
        samples_shuffled = samples[idx]
        batch_losses = []
        num_batches = (N + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, N)
            batch = samples_shuffled[batch_start:batch_end]
            # Use a different rng for each batch
            batch_step_rng = jax.random.fold_in(batch_rng, epoch * num_batches + i)
            loss, grads = jax.value_and_grad(fast_ism_loss)(params, apply_fn, batch, batch_step_rng)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            batch_losses.append(loss)
        avg_loss = np.mean(jax.device_get(jnp.array(batch_losses)))
        losses.append(avg_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, avg loss: {avg_loss:.6f}")
    return params, opt_state, losses
#%%
# SBTM
rng = jax.random.PRNGKey(1)
sample_batch_size = 128
samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
# samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.01
# samples = jnp.zeros((sample_batch_size, 28, 28, 1))

step_size = 0.001
num_steps = 21

current_score = ScoreNetStatic()
static_params = current_score.init(rng, samples)
optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(static_params)
for i in tqdm(range(num_steps)):
    step_rng, rng = jax.random.split(rng)
    samples = samples + step_size * target_score(samples) - current_score.apply(static_params, samples)
    static_params, opt_state, losses = run_fast_ism_gd(static_params, current_score.apply, samples, step_rng, optimizer, opt_state, fast_ism_loss, epochs=10, batch_size=64)
    if i % 1 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()

plt.imshow(gallery(samples, 16))
# %%
