import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
import optax

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)

particles = jnp.array([[-5.0], [0.0], [5.0]])
print("Training data points:", particles.flatten())

from sbtm import models, losses
mlp = models.MLP(d=1, hidden_units=[128, 128, 128])
# score_model = models.ResNet(mlp)
score_model = mlp
optimizer = nnx.Optimizer(score_model, optax.adamw(0.0005, 0.9))

@nnx.jit
def train_step(optimizer, score_model, particles):
    loss_value, grads = nnx.value_and_grad(losses.implicit_score_matching_loss)(score_model, particles)
    optimizer.update(grads)
    return loss_value

x = jnp.linspace(-7, 7, 200).reshape(-1, 1)
plt.figure(figsize=(10, 6))
for i in range(801):
    loss_value = train_step(optimizer, score_model, particles)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss_value}")
        plt.plot(x, score_model(x), label=f"epoch {i} loss={loss_value:.2f}")

plt.legend()
plt.scatter(particles, jnp.zeros_like(particles), color='red', label='particles')
plt.xlabel("x")
plt.ylabel("Score")
plt.savefig("train_implicit.png")

