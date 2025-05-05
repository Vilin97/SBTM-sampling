#%%
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import matplotlib.pyplot as plt
from tqdm import tqdm
from train import ScoreNet, marginal_prob_std

sigma = 25.0
marginal_prob_std_fn = lambda t: jnp.sqrt((sigma**(2 * t) - 1.) / (2. * jnp.log(sigma)))
model = ScoreNet(marginal_prob_std_fn)
params = checkpoints.restore_checkpoint("/mmfs1/gscratch/krishna/psushko/SBTM-sampling/peter/0_1", target=None)['params']

rng = jax.random.PRNGKey(123)
batch = jax.random.normal(rng, (16, 28, 28, 1))
num_steps = 400
step_size = 0.25
ts = jnp.linspace(1.0, 1e-3, num_steps)

for i, t in tqdm(enumerate(ts)):
    rng, step_rng = jax.random.split(rng)
    s = model.apply({'params': params}, batch, jnp.full((batch.shape[0],), t))
    batch += step_size * s / 2
    batch += jnp.sqrt(step_size) * jax.random.normal(step_rng, batch.shape)
    if i % 10 == 0:
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        for k in range(16):
            r, c = divmod(k, 4)
            ax[r, c].imshow(batch[k, ..., 0], cmap='gray')
            ax[r, c].axis('off')
        plt.tight_layout()
        plt.show()

# %%
