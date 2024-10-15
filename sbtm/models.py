from flax import nnx

class MLP(nnx.Module):
    """Muti-Layer Perceptron"""
    def __init__(self, d, hidden_layers = [100], activation = nnx.soft_sign, seed = 0):
        rngs = nnx.Rngs(seed)
        layers = []
        input_dim = d
        for hidden_dim in hidden_layers:
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim
        self.hidden_layers = layers
        self.linear_out = nnx.Linear(input_dim, d, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.linear_out(x)
        return x
    
class ResNet(nnx.Module):
    """Residual Network
    Args:
        mlp (MLP): a multi-layer perceptron
    """
    def __init__(self, mlp):
        self.mlp = mlp

    def __call__(self, x):
        mlp = self.mlp
        for layer in mlp.hidden_layers:
            x = mlp.activation(layer(x)) + x
        x = mlp.linear_out(x)
        return x
