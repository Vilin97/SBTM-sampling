#%% 
import os
import array
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple
import functools
import jax
import gzip
from flax.training import checkpoints
import optax
import flax.training.train_state as train_state
import struct
import urllib.request
from tqdm import tqdm

dir_data = os.path.expanduser(f'~/SBTM-sampling/data')
dir_data_mnist = os.path.join(dir_data, 'mnist')
dir_data_models_mnist = os.path.join(dir_data, 'models/mnist')


def load_mnist():
    filename = "train-images-idx3-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = dir_data_mnist
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"

    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")

    with gzip.open(target, "rb") as fh:
        _, batch, rows, cols = struct.unpack(">IIII", fh.read(16))
        shape = (batch, 1, rows, cols)
        return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8).reshape(shape)


def mnist_labels():
    filename = "train-labels-idx1-ubyte.gz"
    url_dir = "https://storage.googleapis.com/cvdf-datasets/mnist"
    target_dir = dir_data_mnist
    url = f"{url_dir}/{filename}"
    target = f"{target_dir}/{filename}"

    if not os.path.exists(target):
        os.makedirs(target_dir, exist_ok=True)
        urllib.request.urlretrieve(url, target)
        print(f"Downloaded {url} to {target}")

    with gzip.open(target, "rb") as fh:
        _, num_items = struct.unpack(">II", fh.read(8))
        return jnp.array(array.array("B", fh.read()), dtype=jnp.uint8)

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


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The sigma in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return sigma**t


sigma = 25.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


# @jax.jit
def loss_fn(rng, model, params, x, marginal_prob_std, reg=1e-3, t_range=(1e-5, 1.0)):
    """The loss function for training score-based generative models.

    Args:
    model: A `flax.linen.Module` object that represents the structure of
        the score-based model.
    params: A dictionary that contains all trainable parameters.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    rng, step_rng = jax.random.split(rng)
    random_t = jax.random.uniform(step_rng, (x.shape[0],), minval=t_range[0], maxval=t_range[1])
    rng, step_rng = jax.random.split(rng)
    z = jax.random.normal(step_rng, x.shape)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(params, perturbed_x, random_t)
    loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z)**2 + reg * score**2, axis=(1, 2, 3)))
    return loss


def get_train_step_fn(model, marginal_prob_std):
    """Create a one-step training function.

    Args:
    model: A `flax.linen.Module` object that represents the structure of
        the score-based model.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    Returns:
    A function that runs one step of training.
    """
    val_and_grad_fn = jax.value_and_grad(loss_fn, argnums=2)

    def step_fn(rng, x, state):
        loss, grad = val_and_grad_fn(rng, state.apply_fn, state.params, x, marginal_prob_std)
        # mean_grad = jax.lax.pmean(grad, axis_name='device')
        # mean_loss = jax.lax.pmean(loss, axis_name='device')
        return loss, state.apply_gradients(grads=grad)
    return step_fn

# Define dataset
preprocessing = "scaled"   

n_epochs = 151      
batch_size = 256    
lr = 1e-4           

rng = jax.random.PRNGKey(0)
fake_input = jnp.ones((batch_size, 28, 28, 1))
fake_time = jnp.ones(batch_size)
score_model = ScoreNet(marginal_prob_std_fn)
params = score_model.init({'params': rng}, fake_input, fake_time)

data = load_mnist()

sel = jax.random.permutation(rng, len(data))
data = data[sel]

# preprocess data
if preprocessing == "standardized":
    data_mean, data_std = jnp.mean(data), jnp.std(data)
    data = (data - data_mean) / data_std
elif preprocessing == "scaled":
    data = data / data.max()
elif preprocessing is None:
    pass
data = jnp.transpose(a=data, axes=(0, 2, 3, 1))
data = data #+ 1e-4 * jax.random.normal(rng, data.shape)

optimizer = optax.adamw(learning_rate=lr)
train_state_ = train_state.TrainState.create(
    apply_fn=score_model.apply, params=params, tx=optimizer
)
train_step_fn = jax.jit(get_train_step_fn(score_model, marginal_prob_std_fn))

# --- before the training loop ---
final_ckpt_dir = "/mmfs1/gscratch/krishna/psushko/SBTM-sampling/peter/all_digits"
os.makedirs(final_ckpt_dir, exist_ok=True)

# Train
for epoch in tqdm(range(n_epochs), desc='Epochs'):
    # Shuffle data at the beginning of each epoch
    shuffle_rng, rng = jax.random.split(rng)
    shuffle_idx = jax.random.permutation(shuffle_rng, len(data))
    shuffled_data = data[shuffle_idx]
    
    avg_loss = 0.
    num_items = 0
    for i in range(len(data) // batch_size - 1):
        indices = jnp.arange(i * batch_size, (i + 1) * batch_size)
        x = shuffled_data[indices]
        rng, step_rng = jax.random.split(rng)
        step_rng = jnp.asarray(step_rng)
        loss, train_state_ = train_step_fn(step_rng, x, train_state_)
        avg_loss += loss
    
    print('Average Loss: {:5f}'.format(avg_loss / (len(data) // batch_size - 1)))

checkpoints.save_checkpoint(
    ckpt_dir=final_ckpt_dir,
    target=train_state_,
    step=n_epochs,
    keep=1,
    overwrite=True
)

# %%
