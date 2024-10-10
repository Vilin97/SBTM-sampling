import jax
import jax.numpy as jnp
import equinox as eqx

class Kernel(eqx.Module):
    """Base Kernel class."""

    kernel_fun: callable

    def __init__(self, kernel_fun):
        """
        :param kernel_fun: A kernel function with the signature f(X, Y, parameters)
        :type kernel_fun: function
        """
        self.kernel_fun = kernel_fun

    def __call__(self, x, y, width):
        """
        :param x: Input 1
        :type x: array
        :param y: Input 2
        :type y: array
        :return: Gram matrix between x and y with the specified kernel
        :rtype: array
        """
        K = jax.vmap(lambda x1: jax.vmap(lambda y1: self.kernel_fun(x1, y1, width))(y))(x)
        return K

    def gradient_wrt_first_arg(self, x, y, width):
        """
        Compute the gradient of the kernel with respect to the first argument.
        :param x: Input 1
        :type x: array
        :param y: Input 2
        :type y: array
        :return: Gradient of the kernel with respect to the first argument
        :rtype: array
        """
        grad_kernel_fun = jax.grad(self.kernel_fun, argnums=0)
        grad_K = jax.vmap(lambda x1: jax.vmap(lambda y1: grad_kernel_fun(x1, y1, width))(y))(x)
        return grad_K
    
# Example kernel function
def rbf_kernel(x, y, width):
    sqdist = jnp.sum((x - y) ** 2)
    return jnp.exp(-sqdist / width)
