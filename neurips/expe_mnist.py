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
def gallery(array, ncols=16):
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
rng = jax.random.PRNGKey(2)
sample_batch_size = 1024
samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1)) # index 15*32+1 is a '7', index 4 is a '8'
# samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1))# * 0.001
# samples = jnp.zeros((sample_batch_size, 28, 28, 1))

step_size = 0.001
num_steps = 61

for i in tqdm(range(num_steps)):
    step_rng, rng = jax.random.split(rng)
    samples = samples + step_size * target_score(samples)
    samples += jnp.sqrt(2 * step_size) * jax.random.normal(step_rng, samples.shape)
    if i % 10 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(gallery(samples, int(np.sqrt(sample_batch_size))))
#%%
# Langevin
rng = jax.random.PRNGKey(2)
sample_batch_size = 1024
step_size = 0.001
num_steps = 31

samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1)) # index 15*32+1 is a '7', index 4 is a '8'
six = samples[0]
eight = samples[4]
zero = samples[5]
three = samples[7]
two = samples[32]
four = samples[32+2]
five = samples[32+3]
seven = samples[15*32+1]
digits = {'6': six, '8': eight, '0': zero, '3': three, '2': two, '4': four, '5': five, '7': seven}
for (digit, sample) in digits.items():
    samples = sample + jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.000
    for i in tqdm(range(num_steps)):
        step_rng, rng = jax.random.split(rng)
        samples = samples + step_size * target_score(samples)
        samples += jnp.sqrt(2 * step_size) * jax.random.normal(step_rng, samples.shape)
        # if i % 10 == 0:
        #     print(i)
        #     result = gallery(samples[:12])
        #     plt.imshow(result)
        #     plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(gallery(samples, int(np.sqrt(sample_batch_size))))
    plt.axis('off')
    plt.title(f'{digit}')
    plt.show()

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
def train(
    params,
    apply_fn,
    samples,
    rng,
    optimizer,
    opt_state,
    loss_fn,
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
            loss, grads = jax.value_and_grad(loss_fn)(params, apply_fn, batch, batch_step_rng)
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
current_score_model = ScoreNetStatic()
static_params = current_score_model.init(rng, samples)
optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(static_params)

#%%
rng = jax.random.PRNGKey(1)
sample_batch_size = 128
# samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1))
samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.001
# samples = jnp.zeros((sample_batch_size, 28, 28, 1))

# Do a few steps of gradient descent to get current_score_model to output 0 for samples
zero_target = jnp.zeros_like(samples)
gd_epochs = 10

def mse_loss(params, apply_fn, x, _rng):
    pred = apply_fn(params, x)
    return jnp.mean((pred - zero_target[:x.shape[0]]) ** 2)

plt.imshow(gallery(current_score_model.apply(static_params, samples[:12])))
plt.show()

# # Add 10 levels of noise to samples and concatenate them
# num_levels = 10
# noise_levels = jnp.linspace(0.01, 0.1, num_levels)
# samples_aug = jnp.concatenate([
#     samples + jax.random.normal(jax.random.PRNGKey(i), samples.shape) * noise_levels[i]
#     for i in range(num_levels)
# ], axis=0)

static_params, opt_state, loss_history = train(
    static_params,
    current_score_model.apply,
    samples,
    rng,
    optimizer,
    opt_state,
    mse_loss,
    epochs=gd_epochs,
    batch_size=sample_batch_size,
    verbose=True
)

plt.imshow(gallery(current_score_model.apply(static_params, samples[:12])))
plt.show()
plt.plot(loss_history)
plt.show()

#%%
# SBTM sampling
rng = jax.random.PRNGKey(1)
sample_batch_size = 128
samples = jax.random.normal(rng, (sample_batch_size, 28, 28, 1)) * 0.001

step_size = 0.001
num_steps = 41
for i in tqdm(range(num_steps)):
    step_rng, rng = jax.random.split(rng)
    samples = samples + step_size * target_score(samples) - step_size * current_score_model.apply(static_params, samples)
    static_params, opt_state, losses = train(static_params, current_score_model.apply, samples, step_rng, optimizer, opt_state, fast_ism_loss, epochs=10, batch_size=64, verbose=False)
    if i % 5 == 0:
        print(i)
        result = gallery(samples[:12])
        plt.imshow(result)
        plt.show()

plt.imshow(gallery(samples, 16))

# what are the knobs and toggles:
# step_size, num_steps, batch_size, epochs, sample_batch_size
# comapre current_score at the end of the simulation with the ground truth (target_score)

#%%
rng = jax.random.PRNGKey(2)
sample_batch_size = 1024
step_size = 0.001
num_steps = 31

samples = jax.random.uniform(rng, (sample_batch_size, 28, 28, 1)) # index 15*32+1 is a '7', index 4 is a '8'
six = samples[0]
eight = samples[4]
zero = samples[5]
three = samples[7]
two = samples[32]
four = samples[32+2]
five = samples[32+3]
seven = samples[15*32+1]
digits = {'6': six, '8': eight, '0': zero, '3': three, '2': two, '4': four, '5': five, '7': seven}
cosine_sims = {}

small_batch_size = 128
for (digit, sample) in digits.items():
    samples = sample + jax.random.normal(rng, (small_batch_size, 28, 28, 1)) * 0.01
    current_score_model = ScoreNetStatic()
    static_params = current_score_model.init(rng, samples)
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(static_params)
    
    # set the first few samples to other modes
    samples = samples.at[0].set(zero).at[1].set(two).at[2].set(three).at[3].set(four).at[4].set(five).at[5].set(six).at[6].set(seven).at[7].set(eight)
    
    cosine_sims[digit] = []
    for i in tqdm(range(num_steps)):
        step_rng, rng = jax.random.split(rng)
        ts = target_score(samples)
        cs = current_score_model.apply(static_params, samples)
        samples = samples + step_size * ts - step_size * cs
        static_params, opt_state, losses = train(static_params, current_score_model.apply, samples, step_rng, optimizer, opt_state, fast_ism_loss, epochs=2, batch_size=64, verbose=False)
            
        # Compute cosine similarity between target_score and current_score
        ts_flat = ts.reshape(ts.shape[0], -1)
        cs_flat = cs.reshape(cs.shape[0], -1)
        dot_product = jnp.sum(ts_flat * cs_flat, axis=1)
        ts_norm = jnp.linalg.norm(ts_flat, axis=1)
        cs_norm = jnp.linalg.norm(cs_flat, axis=1)
        cosine_sim = dot_product / (ts_norm * cs_norm + 1e-8)
        avg_cosine_sim = jnp.mean(cosine_sim)
        cosine_sims[digit].append(float(avg_cosine_sim))
        # if i % 2 == 0:
        #     print(f"Cosine similarity at step {i}: {avg_cosine_sim:.4f}")
        
        # if i % 10 == 0:
        #     print(i)
        #     result = gallery(samples[:12])
        #     plt.imshow(result)
        #     plt.show()
            
    # Plot the cosine similarity
    plt.figure(figsize=(8, 2))
    plt.plot(cosine_sims[digit])
    plt.title(f'Cosine similarity (digit {digit})')
    plt.xlabel('Step')
    plt.ylabel('Cosine similarity')
    plt.show()

    # Plot the final samples
    plt.figure(figsize=(8, 8))
    plt.imshow(gallery(samples, 16))
    plt.axis('off')
    plt.title(f'SBTM {digit}')
    plt.show()
    
    # Compare to gradient ascent
    samples = sample + jax.random.normal(rng, (small_batch_size, 28, 28, 1)) * 0.01
    samples = samples.at[0].set(zero).at[1].set(two).at[2].set(three).at[3].set(four).at[4].set(five).at[5].set(six).at[6].set(seven).at[7].set(eight)
    for i in tqdm(range(num_steps)):
        step_rng, rng = jax.random.split(rng)
        ts = target_score(samples)
        cs = current_score_model.apply(static_params, samples)
        samples = samples + step_size * ts
            
    # Plot the final samples
    plt.figure(figsize=(8, 8))
    plt.imshow(gallery(samples, 16))
    plt.axis('off')
    plt.title(f'Gradient Ascent {digit}')
    plt.show()
    
#%%
rng = jax.random.PRNGKey(2)
step_size = 0.001
num_steps = 31

small_batch_size = 64
samples = jax.random.uniform(rng, (small_batch_size, 28, 28, 1))
other_samples = jax.random.uniform(jax.random.PRNGKey(2), (small_batch_size, 28, 28, 1))
other_samples2 = jax.random.uniform(jax.random.PRNGKey(2), (small_batch_size, 28, 28, 1))
other_samples3 = jax.random.uniform(jax.random.PRNGKey(2), (small_batch_size, 28, 28, 1))

current_score_model = ScoreNetStatic()
static_params = current_score_model.init(rng, samples)
optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(static_params)

cosine_sims = []
other_cosine_sims = []
for i in tqdm(range(num_steps)):
    noise_factor = 20*(num_steps/(20*i+1))
    step_rng, rng = jax.random.split(rng)
    ts = target_score(samples)
    cs = current_score_model.apply(static_params, samples)
    samples = samples + step_size * ts - noise_factor * step_size * cs
    static_params, opt_state, losses = train(static_params, current_score_model.apply, samples, step_rng, optimizer, opt_state, fast_ism_loss, epochs=10, batch_size=64, verbose=False)
    print(jnp.mean(jnp.abs(ts)))
    print(jnp.mean(jnp.abs(cs)))
    print(jnp.mean(jnp.abs(samples)))
        
    other_samples = other_samples + step_size * target_score(other_samples)
    other_samples2 = other_samples2 + step_size * target_score(other_samples2) + step_size * jnp.sqrt(2 * step_size) * jax.random.normal(step_rng, other_samples2.shape)
    other_samples3 = other_samples3 + step_size * target_score(other_samples3) + jnp.sqrt(2 * noise_factor * step_size) * jax.random.normal(step_rng, other_samples3.shape)
    
    # Compute cosine similarity between target_score and current_score
    ts_flat = ts.reshape(ts.shape[0], -1)
    cs_flat = cs.reshape(cs.shape[0], -1)
    dot_product = jnp.sum(ts_flat * cs_flat, axis=1)
    ts_norm = jnp.linalg.norm(ts_flat, axis=1)
    cs_norm = jnp.linalg.norm(cs_flat, axis=1)
    cosine_sim = dot_product / (ts_norm * cs_norm + 1e-8)
    avg_cosine_sim = jnp.mean(cosine_sim)
    cosine_sims.append(float(avg_cosine_sim))
    
    ts = target_score(other_samples)
    cs = current_score_model.apply(static_params, other_samples)
    other_ts_flat = ts.reshape(ts.shape[0], -1)
    other_cs_flat = cs.reshape(cs.shape[0], -1)
    other_dot_product = jnp.sum(other_ts_flat * other_cs_flat, axis=1)
    other_ts_norm = jnp.linalg.norm(other_ts_flat, axis=1)
    other_cs_norm = jnp.linalg.norm(other_cs_flat, axis=1)
    other_cosine_sim = other_dot_product / (other_ts_norm * other_cs_norm + 1e-8)
    avg_other_cosine_sim = jnp.mean(other_cosine_sim)
    other_cosine_sims.append(float(avg_other_cosine_sim))
    
plt.figure(figsize=(8, 2))
plt.plot(cosine_sims, label='Same')
plt.plot(other_cosine_sims, label='GA')
plt.title(f'Cosine similarity')
plt.xlabel('Step')
plt.ylabel('Cosine similarity')
plt.legend()
plt.show()

# %%
plt.imshow(gallery(samples, 16))
plt.axis('off')
plt.title(f'SBTM with scaled noise')
plt.show()

plt.imshow(gallery(other_samples, 16))
plt.axis('off')
plt.title(f'GA')
plt.show()

plt.imshow(gallery(other_samples2, 16))
plt.axis('off')
plt.title(f'Langevin')
plt.show()

plt.imshow(gallery(other_samples3, 16))
plt.axis('off')
plt.title(f'Langevin with scaled noise')
plt.show()

#%%
cosine_sims = {}
#%%
epochs = 30
rng = jax.random.PRNGKey(2)
step_size = 0.001
num_steps = 31

small_batch_size = 64
samples = jax.random.normal(rng, (small_batch_size, 28, 28, 1))
samples2 = jax.random.normal(rng, (small_batch_size, 28, 28, 1))
other_samples = jax.random.normal(jax.random.PRNGKey(2), (small_batch_size, 28, 28, 1))
other_samples2 = jax.random.normal(jax.random.PRNGKey(2), (small_batch_size, 28, 28, 1))
other_samples3 = jax.random.normal(jax.random.PRNGKey(2), (small_batch_size, 28, 28, 1))

current_score_model = ScoreNetStatic()
static_params = current_score_model.init(rng, samples)
optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(static_params)

current_score_model2 = ScoreNetStatic()
static_params2 = current_score_model2.init(rng, samples2)
optimizer2 = optax.adamw(1e-3)
opt_state2 = optimizer2.init(static_params2)

cosine_sims[epochs] = []
for i in tqdm(range(num_steps)):
    noise_factor = 10*(num_steps/(20*i+1))
    step_rng, rng = jax.random.split(rng)
    ts = target_score(samples)
    cs = current_score_model.apply(static_params, samples)
    samples = samples + step_size * ts - noise_factor * step_size * cs
    static_params, opt_state, losses = train(static_params, current_score_model.apply, samples, step_rng, optimizer, opt_state, fast_ism_loss, epochs=epochs, batch_size=64, verbose=False)
    
    cs2 = current_score_model2.apply(static_params2, samples2)
    samples2 = samples2 + step_size * ts - step_size * cs
    static_params2, opt_state2, losses = train(static_params2, current_score_model2.apply, samples2, step_rng, optimizer2, opt_state2, fast_ism_loss, epochs=epochs, batch_size=64, verbose=False)
        
    other_samples = other_samples + step_size * target_score(other_samples)
    other_samples2 = other_samples2 + step_size * target_score(other_samples2) + step_size * jnp.sqrt(2 * step_size) * jax.random.normal(step_rng, other_samples2.shape)
    other_samples3 = other_samples3 + step_size * target_score(other_samples3) + jnp.sqrt(2 * noise_factor * step_size) * jax.random.normal(step_rng, other_samples3.shape)
    
    # Compute cosine similarity between target_score and current_score
    ts_flat = ts.reshape(ts.shape[0], -1)
    cs_flat = cs.reshape(cs.shape[0], -1)
    dot_product = jnp.sum(ts_flat * cs_flat, axis=1)
    ts_norm = jnp.linalg.norm(ts_flat, axis=1)
    cs_norm = jnp.linalg.norm(cs_flat, axis=1)
    cosine_sim = dot_product / (ts_norm * cs_norm + 1e-8)
    avg_cosine_sim = jnp.mean(cosine_sim)
    cosine_sims[epochs].append(float(avg_cosine_sim))
    
# %%
plt.imshow(gallery(samples2, 16))
plt.axis('off')
plt.title(f'SBTM')
plt.show()

plt.imshow(gallery(samples, 16))
plt.axis('off')
plt.title(f'SBTM with scaled noise')
plt.show()

# plt.imshow(gallery(other_samples, 16))
# plt.axis('off')
# plt.title(f'GA')
# plt.show()

# plt.imshow(gallery(other_samples2, 16))
# plt.axis('off')
# plt.title(f'Langevin')
# plt.show()

# plt.imshow(gallery(other_samples3, 16))
# plt.axis('off')
# plt.title(f'Langevin with scaled noise')
# plt.show()
#%%
plt.figure(figsize=(8, 2))
for epoch in [2,5,10,30,100]:
    plt.plot([0.001*t for t in range(num_steps)], cosine_sims[epoch], label=f'epoch {epoch}')
plt.title(f'Cosine similarity')
plt.xlabel('time')
plt.ylabel('Cosine similarity')
plt.legend()
plt.show()
   
#%%
rng = jax.random.PRNGKey(2)
step_size = 0.001
num_steps = 31

small_batch_size = 64
samples = jax.random.uniform(rng, (small_batch_size, 28, 28, 1))
other_samples = jax.random.uniform(jax.random.PRNGKey(3), (small_batch_size, 28, 28, 1))

current_score_model = ScoreNetStatic()
static_params = current_score_model.init(rng, samples)
optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(static_params)

cosine_sims = []
for i in tqdm(range(num_steps)):
    step_rng, rng = jax.random.split(rng)
    ts = target_score(samples)
    samples = samples + step_size * ts + jnp.sqrt(2 * step_size) * jax.random.normal(step_rng, samples.shape)
            
    other_ts = target_score(other_samples)
    other_samples = other_samples + step_size * other_ts + jnp.sqrt(2 * step_size) * jax.random.normal(step_rng, other_samples.shape)
    
    ts_flat = ts.reshape(ts.shape[0], -1)
    other_ts_flat = other_ts.reshape(cs.shape[0], -1)
    dot_product = jnp.sum(ts_flat * other_ts_flat, axis=1)
    ts_norm = jnp.linalg.norm(ts_flat, axis=1)
    cs_norm = jnp.linalg.norm(other_ts_flat, axis=1)
    cosine_sim = dot_product / (ts_norm * cs_norm + 1e-8)
    avg_cosine_sim = jnp.mean(cosine_sim)
    cosine_sims.append(float(avg_cosine_sim))
    
    
plt.imshow(gallery(samples, 16))
plt.axis('off')
plt.show()
plt.imshow(gallery(other_samples, 16))
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 2))
plt.plot(cosine_sims, label='Same')
plt.title(f'Cosine similarity')
plt.xlabel('Step')
plt.ylabel('Cosine similarity')
plt.legend()
plt.show()


#%%
samples = jax.random.uniform(rng, (small_batch_size, 28, 28, 1))
score = current_score_model.apply(static_params, samples)
plt.imshow(gallery(samples))
plt.show()
plt.imshow(gallery(score))
plt.show()

plt.imshow(gallery(samples + score))
plt.show()