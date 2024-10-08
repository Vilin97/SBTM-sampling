"""Project skeleton"""

"""
What do I want from the code?
- compare different samplers: SBTM, SVGD, Langevin dynamics
- compare different NN architectures for SBTM: MLP, ResNet, transformer
- compare different numbers of SGD steps for SBTM
- compare different loss functions for SBTM

- pre-train initial NNs and save them
- save results and intermediate states and run analysis afterwards
- compute and plot various metrics: Fisher divergence, KL divergence, moments, score approximation error, etc.
- record lots of things: particle positions, NN weights, etc.

- everything should be on the GPU
- use Flax for NNs
"""
import jax.numpy as jnp
import jax.random as jrandom

class Logger:
    def log(self, **kwargs):
        pass

# Main class to sample
class Sampler:
    def __init__(self, particles, target_score, step_size, max_steps, logger):
        self.particles = particles  # jnp array of particles
        self.target_score = target_score  # target score function
        self.step_size = step_size  # delta t
        self.max_steps = max_steps  # maximum number of steps
        self.logger = logger  # logger object

    def sample(self):
        """Sample from the target distribution"""
        for step_number in range(self.max_steps):
            self.step()
            self.logger.log(particles=self.particles, step_number=step_number, score=self.target_score(self.particles))

        return self.particles

    def step(self):
        """Take one step, e.g. predict score and move particles"""
        raise NotImplementedError("must be implemented by subclasses")


class SDESampler(Sampler):
    """Langevin dynamics sampler"""

    def __init__(self, particles, target_score, step_size, max_steps, logger, seed=0):
        super().__init__(particles, target_score, step_size, max_steps, logger)
        self.key = jrandom.PRNGKey(seed)

    def step(self):
        """Draw brownian noise Z and move particles in direction target_score - sqrt(2)Z"""
        dim = self.particles.shape[1]
        noise = jrandom.multivariate_normal(self.key, jnp.zeros_like(self.particles), jnp.eye(dim))
        drift = self.target_score(self.particles)
        self.particles += self.step_size * drift + jnp.sqrt(2 * self.step_size) * noise




class ODESampler(Sampler):
    """Deterministic sampler"""
    def __init__(self, particles, target_score, step_size, max_steps, logger, score_model):
        super().__init__(particles, target_score, step_size, max_steps, logger)
        self.score_model = score_model  # a model to approximate grad-log-density, e.g. a NN

    def step(self):
        """Train the score model and move particles in direction target_score - score_model(particles)"""
        pass


class SBTMSampler(ODESampler):
    """Use a NN to approximate the score"""
    # assume that the score model has already been pre-trained

class SVGDSampler(ODESampler):
    """Use RKHS to approximate the score"""

# Logger class to log target score, step_size, max-steps, step number, particle locations, predicted score, score model. Should have method `log`.
class Logger:
    def log(self, **kwargs):
        pass

# ScoreModel class to approximate the score function. It should have methods `fit` and `predict`. `predict` is the forward pass of the model. `fit` should fit the model given particle locations. The class is inherited by NNScoreModel and RKHSScoreModel. NNScoreModel should contain a Flax model, a loss, an optimizer, and a callable `stop_gd` that stops the gradient descent (it can perform a fixed number of GD steps or implement a more sophisticated stopping criterion). RKHSScoreModel should contain the kernel function.
class ScoreModel:
    def fit(self, particles):
        raise NotImplementedError("must be implemented by subclasses")

    def predict(self, particles):
        raise NotImplementedError("must be implemented by subclasses")


class NNScoreModel(ScoreModel):
    def __init__(self, model, loss, optimizer, stop_gd):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.stop_gd = stop_gd

    def fit(self, particles):
        pass

    def predict(self, particles):
        pass


class RKHSScoreModel(ScoreModel):
    def __init__(self, kernel_function):
        self.kernel_function = kernel_function

    def fit(self, particles):
        pass

    def predict(self, particles):
        pass

# Experiment class to orchestrate running a single experiment. It should take a `config` dict. It should have methods `run` and `compute_metrics`. `run` should load the model (or train a new one if not available), init the sampler, run the sampler, and save the config and the results. `compute_metrics` should load the results, compute metrics and save the metrics. `plot_metrics` should load the metrics, plot them, and save the plots.
class Experiment:
    def __init__(self, config):
        self.config = config

    def run(self):
        pass

    def compute_metrics(self):
        pass

    def plot_metrics(self):
        pass


# DataManager class to save and load data. Should save and load experiment data and plots. Not sure how to imlpement this yet. Maybe with pickle or maybe with a PostgreSQL database or maybe with pandas. Or maybe with Weights and Biases!
class DataManager:
    def save_data(self, data, filename):
        pass

    def load_data(self, filename):
        pass

    def save_plot(self, plot, filename):
        pass

    def load_plot(self, filename):
        pass
