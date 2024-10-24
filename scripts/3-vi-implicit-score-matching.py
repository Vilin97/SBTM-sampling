# See if the NN will overfit using implicit score matching

#%%
import importlib
import jax
import jax.numpy as jnp
import jax.random as jrandom
from sbtm import density, plots, kernel, losses, models, sampler
from flax import nnx
import optax
import os
import copy
os.environ["JAX_CHECK_TRACER_LEAKS"] = 'True'
import matplotlib.pyplot as plt

for module in [density, plots, kernel, losses, models, sampler]:
    importlib.reload(module)
    
    
# %%
num_particles = 100
key = jrandom.key(42)

prior_params = {'mean': jnp.array([0.]), 'covariance': jnp.array([[1.]])}
prior_sample = jrandom.multivariate_normal(key, prior_params['mean'], prior_params['covariance'], shape=(num_particles,))
prior_score = density.Density(density.gaussian_pdf, prior_params).score

target_params = {'mean': jnp.array([0.]), 'covariance': jnp.array([[1.]])}
target_density_obj = density.Density(density.gaussian_pdf, target_params)
target_score = target_density_obj.score

#%%
# Compare explicit and implicit score matching
mlp = models.MLP(d=1)
score_model_1 = models.ResNet(mlp)
score_model_2 = copy.deepcopy(score_model_1)

for (score_model, loss) in zip([score_model_1, score_model_2], [losses.explicit_score_matching_loss, losses.implicit_score_matching_loss]):
    optimizer = nnx.Optimizer(score_model, optax.adamw(0.001))
    for i in range(101):
        if i % 10 == 0:
            explicit_loss_value = losses.explicit_score_matching_loss(score_model, prior_score, prior_sample)
            implicit_loss_value = losses.implicit_score_matching_loss(score_model, prior_sample)
            print(f"Iteration {i}: Explicit Loss = {explicit_loss_value:.5f}, Implicit Loss = {implicit_loss_value:.2f}")
        
        if loss == losses.implicit_score_matching_loss:
            loss_value, grads = nnx.value_and_grad(loss)(score_model, prior_sample)
        else:
            loss_value, grads = nnx.value_and_grad(loss)(score_model, prior_score, prior_sample)
        optimizer.update(grads)

    x = jnp.linspace(-5, 5, 1000).reshape(-1, 1)
    score_model_output = score_model(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, score_model_output, label='Score Model')
    plt.scatter(prior_sample, jnp.zeros_like(prior_sample), color='red', label='Prior Samples', s=10)
    plt.xlabel('x')
    plt.ylabel('Score')
    plt.title('Score Model Output')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
# Compare MLP and ResNet architectures. Resnet can approximate -x much better than MLP
mlp = models.MLP(d=1)
score_model_1 = mlp
score_model_2 = models.ResNet(copy.deepcopy(mlp))

for score_model in [score_model_1, score_model_2]:
    optimizer = nnx.Optimizer(score_model, optax.adamw(0.001))
    for i in range(101):
        if i % 10 == 0:
            explicit_loss_value = losses.explicit_score_matching_loss(score_model, prior_score, prior_sample)
            print(f"Iteration {i}: Explicit Loss = {explicit_loss_value:.5f}")
        
        loss_value, grads = nnx.value_and_grad(losses.explicit_score_matching_loss)(score_model, prior_score, prior_sample)
        optimizer.update(grads)

    x = jnp.linspace(-5, 5, 1000).reshape(-1, 1)
    score_model_output = score_model(x)

    model_name = type(score_model).__name__
    plt.figure(figsize=(10, 6))
    plt.plot(x, score_model_output, label='Score Model')
    plt.scatter(prior_sample, jnp.zeros_like(prior_sample), color='red', label='Prior Samples', s=10)
    plt.xlabel('x')
    plt.ylabel('Score')
    plt.title(f'{model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

