from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os

def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path + '/state', state)

def load_model(model_reference, path):
    path += '/state'
    graphdef, abstract_state = nnx.split(model_reference)
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(path, abstract_state)

    model = nnx.merge(graphdef, state_restored)
    return model

class MLP(nnx.Module):
    """Muti-Layer Perceptron"""
    def __init__(self, d, hidden_units = [128, 128], activation = nnx.soft_sign, seed = 0):
        rngs = nnx.Rngs(seed)
        layers = []
        input_dim = d
        for hidden_dim in hidden_units:
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim
        self.hidden_units = layers
        self.linear_out = nnx.Linear(input_dim, d, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        for layer in self.hidden_units:
            x = self.activation(layer(x))
        x = self.linear_out(x)
        return x
    
class ResNet(nnx.Module):
    """Residual Network
    Args:
        mlp (MLP): a multi-layer perceptron
    """
    def __init__(self, mlp, seed=0):
        self.mlp = mlp
        rngs = nnx.Rngs(seed)

        # Create a list of projections, one per hidden layer
        self.projections = []
        for layer in mlp.hidden_units:
            in_dim, out_dim = layer.kernel.shape
            # If input dim doesn't match output dim of this layer, create a projection
            if in_dim != out_dim:
                layer = nnx.Linear(in_dim, out_dim, rngs=rngs)
                layer.kernel = nnx.Param(jnp.ones((in_dim, out_dim)))
                self.projections.append(layer)
            else:
                self.projections.append(None)

    def __call__(self, x):
        for layer, projection in zip(self.mlp.hidden_units, self.projections):
            if projection is not None:
                # Project residual to match layer's output dimension
                x = self.mlp.activation(layer(x)) + projection(x)
            else:
                x = self.mlp.activation(layer(x)) + x
        x = self.mlp.linear_out(x)
        return x