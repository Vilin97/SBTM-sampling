from flax import nnx
import jax.numpy as jnp

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
                self.projections.append(nnx.Linear(in_dim, out_dim, rngs=rngs))
            else:
                self.projections.append(None)

    def __call__(self, x):
        for layer, projection in zip(self.mlp.hidden_units, self.projections):
            residual = x
            if projection is not None:
                # Project residual to match layer's output dimension
                residual = projection(residual)
            x = self.mlp.activation(layer(x)) + residual
        x = self.mlp.linear_out(x)
        return x