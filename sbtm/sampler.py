import jax.numpy as jnp
import jax.random as jrandom
import jax
import equinox as eqx
from copy import deepcopy
from tqdm import tqdm


class Logger:
    """Logger class to log target score, step_size, max-steps, step number, particle locations, predicted score, score model. Should have method `log`."""

    def __init__(self):
        self.logs = []

    def log(self, **kwargs):
        self.logs.append(kwargs)


class Sampler:
    particles: jnp.array   # transported particles
    target_score: callable # target score function
    step_size: float       # time discretization
    max_steps: int         # maximum number of steps
    logger: Logger         # logger object

    def __init__(self, particles, target_score, step_size, max_steps, logger):
        self.particles = particles
        self.target_score = target_score  
        self.step_size = step_size
        self.max_steps = max_steps
        self.logger = logger

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

class SBTMSampler(Sampler):
    """Use a NN to approximate the score"""

    def __init__(self, particles, target_score, step_size, max_steps, logger, score_model):
        super().__init__(particles, target_score, step_size, max_steps, logger)
        self.score_model = score_model  # a model to approximate grad-log-density, e.g. a NN
    
    # assume that the score model has already been pre-trained
    # TODO: implement SBTM


class SVGDSampler(ODESampler):
    """Use RKHS to approximate the velocity (target_score - score)"""

    kernel_obj: eqx.Module

    def __init__(self, kernel):
        self.kernel_obj = kernel

    @eqx.filter_jit
    def calculate_gradient(self, score_obj, particles, kernel_params=None):
        num_particles = particles.shape[0]
        gram_matrix = self.kernel_obj(particles, particles, kernel_params)
        score_matrix = score_obj(particles)
        kernel_gradient = self.kernel_obj.gradient_wrt_first_arg(particles,
                                                                 particles,
                                                                 kernel_params)

        # calculate the terms
        kernel_score_term = jnp.einsum('ij,jk->ik', gram_matrix, score_matrix)
        kernel_gradient_term = jnp.sum(kernel_gradient, axis=0)
        phi = (kernel_score_term + kernel_gradient_term) / num_particles
        return phi
    
    @eqx.filter_jit
    def calculate_length_scale(self, particles):
        pairwise_sq_distances= jax.vmap(
            lambda x1: jax.vmap(lambda y1: jnp.sum((x1 - y1) ** 2))(
                particles))(particles)
        median_distance = jnp.sqrt(jnp.median(pairwise_sq_distances))
        new_length_scale = median_distance**2 / jnp.log(len(particles))
        return new_length_scale

    @eqx.filter_jit
    def update(self, particles, score_obj, step_size, kernel_params=None):
        gradient = self.calculate_gradient(score_obj, particles, kernel_params)
        # print(f'Gradient: {jnp.linalg.norm(gradient)}')
        updated_particles = particles + step_size * gradient
        return updated_particles

    def predict(self, particles, score_obj, num_iterations, step_size,
                trajectory=False, adapt_length_scale=False):
        particle_trajectory = np.zeros((num_iterations + 1, particles.shape[0],
                              particles.shape[1]))
        start = deepcopy(particles)
        particle_trajectory[0] = start
        kernel_params = self.kernel_obj.params # initialize

        for i in tqdm(range(1, num_iterations + 1)):
            if 'length_scale' in kernel_params and adapt_length_scale:
                # Update length scale outside of JIT
                kernel_params['length_scale'] = self.calculate_length_scale(start)
            # print(f'Kernel params: {kernel_params}')
            start = self.update(start, score_obj, step_size, kernel_params)
            particle_trajectory[i] = start

        if trajectory:
            return particle_trajectory[-1], particle_trajectory

        else:
            return particle_trajectory[-1]

    # TODO: write the step method
