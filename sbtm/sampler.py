import jax.numpy as jnp
import jax.random as jrandom

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
        self.key = jrandom.key(seed)

    def step(self):
        """Equation (4) in https://arxiv.org/pdf/1907.05600"""
        self.key, subkey = jrandom.split(self.key)
        n, dim = self.particles.shape
        noise = jrandom.multivariate_normal(subkey, jnp.zeros((n,dim)), jnp.eye(dim))
        drift = self.target_score(self.particles)
        self.particles += self.step_size * drift + jnp.sqrt(2 * self.step_size) * noise


class ODESampler(Sampler):
    """Deterministic sampler"""

    def __init__(self, particles, target_score, step_size, max_steps, logger, score_model):
        super().__init__(particles, target_score, step_size, max_steps, logger)
        self.score_model = score_model  # a model to approximate grad-log-density, e.g. a NN

    def step(self):
        """Train the score model and move particles in direction target_score - score_model(particles)"""
        raise NotImplementedError("must be implemented by subclasses")


class SBTMSampler(ODESampler):
    """Use a NN to approximate the score"""

    # assume that the score model has already been pre-trained
    # TODO: implement SBTM


class SVGDSampler(ODESampler):
    """Use RKHS to approximate the velocity (target_score - score)"""

    # TODO: implement SVGD


class Logger:
    """Logger class to log target score, step_size, max-steps, step number, particle locations, predicted score, score model. Should have method `log`."""

    def __init__(self):
        self.logs = []

    def log(self, **kwargs):
        self.logs.append(kwargs)


