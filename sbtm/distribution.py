from jax import grad
from jax.scipy.stats import multivariate_normal
import jax.random as jrandom
import jax.numpy as jnp

class Distribution:
    def __init__(self):
        pass

    def log_density(self, x):
        """Compute the log_density of the distribution at x."""
        log_density = jnp.log(jnp.clip(self.density(x), a_min=1e-10))
        return log_density
    
    def __call__(self, x):
        """Alias for the density method."""
        return self.density(x)

    def score(self, x):
        """Compute the score (gradient of log_density)."""
        return grad(self.log_density)(x)

    def density(self, x):
        """Compute the density of the distribution at x."""
        raise NotImplementedError("Subclasses must implement the density method.")

    def sample(self, key, size=1):
        """Generate samples from the distribution."""
        raise NotImplementedError("Subclasses must implement the sample method.")

class GaussianMixture(Distribution):
    def __init__(self, means, covariances, weights):
        self.means = [jnp.array(mean, ndmin=1) for mean in means]
        self.dimension = len(self.means[0])
        self.covariances = [jnp.eye(self.dimension) * covariance for covariance in covariances]
        self.weights = jnp.array(weights)
        self.num_components = len(weights)

    def density(self, x):
        densities = jnp.array([multivariate_normal.pdf(x, self.means[i], self.covariances[i])
                    for i in range(self.num_components)])
        return jnp.dot(self.weights, densities)

    def sample(self, key, size=1):
        component_indices = jrandom.choice(key, self.num_components, shape=(size,), p=self.weights)
        keys = jrandom.split(key, size)
        samples = jnp.array([self._sample_component(keys[i], component_indices[i]) for i in range(size)])
        return samples

    def _sample_component(self, key, component_index):
        mean = self.means[component_index]
        covariance = self.covariances[component_index]
        return jrandom.multivariate_normal(key, mean, covariance)
    
class Circle(Distribution):
    def __init__(self, center, radius, noise):
        self.center = jnp.array(center)
        self.radius = radius
        self.noise = noise
        self.dimension = len(center)

    def log_density(self, x):
        assert x.shape[1] == self.dimension
        return multivariate_normal.logpdf(jnp.linalg.norm(x-self.center, axis=1) - self.radius, 0., self.noise)
    
    def density(self, x):
        return jnp.exp(self.log_density(x))
    
    def sample(self, key, size=1):
        angles = jrandom.uniform(key, shape=(size,)) * 2 * jnp.pi
        circle_points = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1) * self.radius
        noise = jrandom.normal(key, shape=(size, self.dimension)) * self.noise
        samples = self.center + circle_points + noise
        return samples