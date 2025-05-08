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
import tqdm
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
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
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


def neg_score_fn(x, score_model, params, t):
    t = jnp.ones(x.shape[0],) * t
    return -score_model.apply(params, x, t)


F = functools.partial(neg_score_fn, score_model=score_model, params=params, t=0.001)


def target_score(x):
    """Computes the score of the target distribution.
    Args:
      x: A batch of samples.
         shape = (batch_size, 28 * 28 * 1)
    """
    x = x.reshape((x.shape[0], 28, 28, 1))
    t = jnp.ones(x.shape[0],) * 0.001
    score_val = score_model.apply(params, x, t).reshape(x.shape[0], 28 * 28)
    return score_val


# %%
# Optimization
sample_batch_size = 128
# samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.01
# samples = jnp.zeros((sample_batch_size, 28, 28, 1))

# samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1), minval=-0.42407128, maxval=2.821526)
for i in range(81):
    if i % 20 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()
    # score = score_fn(score_model, params, samples, time)
    samples = samples - 0.001 * F(samples)

# %%
# Visualization

x_grid_ula_uniform = einops.rearrange(
    samples[:100, ..., 0],
    "(n1 n2) (h) (w) -> (n1 h) (n2 w)",
    n1=10, n2=10, h=28, w=28
)
plt.axis("off")
plt.imshow(x_grid_ula_uniform, cmap="Greys")

# %%
# Vanilla Langevin Sampling
sample_batch_size = 100  # 100
n_iter = 300

# Initialization plays a big role
# Uniform
# samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
# # Gaussian
# samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1))
# Dirac
# samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
samples = jnp.zeros((sample_batch_size, 28, 28, 1))


time = jnp.ones((sample_batch_size, ))
tau = 1e-3
temperature = 1
time = jnp.ones((sample_batch_size, )) * 0.001
for i in range(n_iter):
    if i % 20 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()
    step_rng, rng = jax.random.split(rng)
    samples = samples - tau * F(samples) / temperature + jnp.sqrt(2 * tau) * jax.random.normal(step_rng, samples.shape)

#%%
# TODO: need to adjust the time step to be smaller when t is close to 0, otherwise get NaNs
# Omar's dilation path
sample_batch_size = 100  # 100
n_iter = 300
samples = jnp.zeros((sample_batch_size, 28, 28, 1))

time = jnp.ones((sample_batch_size, ))
tau = 1e-3
temperature = 1
time = jnp.ones((sample_batch_size, )) * 0.001
for i in range(n_iter):
    t = i / n_iter
    if i % 20 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()
    step_rng, rng = jax.random.split(rng)
    samples = samples - tau * t * F(samples / t) / temperature + jnp.sqrt(2 * tau) * jax.random.normal(step_rng, samples.shape)

#%%
# NOTE: using my own Langevin sampler
n = 6
m = 6
batch = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
num_steps = 500
step_size = 0.001
ts = jnp.linspace(0, num_steps*step_size, num_steps)
for (ti,t) in tqdm.tqdm(enumerate(ts)):
    step_rng, rng = jax.random.split(rng)
    s = score_model.apply(params, batch, jnp.zeros(batch.shape[0])+1e-3)
    batch += step_size * s / 2
    batch += jnp.sqrt(step_size) * jax.random.normal(step_rng, batch.shape)
    if ti % 20 == 0:
        fig, axes = plt.subplots(n, m, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Reduce space between subplots
        for i in range(min(n*m, batch.shape[0])):
            row, col = i // n, i % m
            im = axes[row, col].imshow(batch[i].squeeze(), cmap='gray')
            axes[row, col].axis('off')
        
        # Add a single colorbar for the entire grid
        plt.tight_layout()
        plt.show()

# %%
# Visualization

x_grid_ula_uniform = einops.rearrange(
    samples[:, ..., 0],
    "(n1 n2) (h) (w) -> (n1 h) (n2 w)",
    n1=10, n2=10, h=28, w=28
)
plt.axis("off")
plt.imshow(x_grid_ula_uniform, cmap="Greys")
