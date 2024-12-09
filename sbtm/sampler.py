import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from tqdm import tqdm
import optax
from flax import nnx


class Logger:
    """Logger class to log target score, step_sizes, max-steps, step number, particle locations, predicted score, score model. Should have method `log`."""
    def __init__(self):
        self.logs = []
        self.hyperparameters = {}

    def log(self, to_log):
        self.logs.append(to_log)
        
    def get_trajectory(self, key):
        return [log[key] for log in self.logs]
    
    def log_hyperparameters(self, hyperparameters):
        self.hyperparameters.update(hyperparameters)

class State:
    def __init__(self, particles):
        self.particles = particles
        self.t = 0
        self.step_number = 0

    def update(self, particles, step_size):
        self.particles = particles
        self.t += step_size
        self.step_number += 1

class Sampler:
    state: State  # state object
    target_score: callable  # target score function
    step_sizes: float  # time discretization
    max_steps: int  # maximum number of steps
    logger: Logger  # logger object
    t_end: float  # end time

    def __init__(self, particles, target_score, step_sizes, max_steps, logger, t_end=None):
        self.state = State(particles)
        try:
            target_score(particles)
            self.target_score = lambda t, x: target_score(x)
        except TypeError:
            self.target_score = target_score
        self.step_sizes = [step_sizes] * max_steps if isinstance(step_sizes, (int, float)) else step_sizes
        self.max_steps = max_steps
        self.logger = logger
        self.logger.log_hyperparameters({'step_sizes': self.step_sizes, 'max_steps': self.max_steps})
        if t_end:
            self.t_end = t_end
        elif max_steps and step_sizes:
            self.t_end = max_steps * jnp.sum(step_sizes)
        else:
            self.t_end = jnp.inf

    def sample(self):
        """Sample from the target distribution"""
        for step_number in tqdm(range(self.max_steps), desc="Sampling"):
            if step_number == self.max_steps or self.state.t >= self.t_end:
                break
            step_size = self.step_size(step_number)
            to_log = {'particles': self.state.particles, 'step_number': step_number, 'step_size': step_size, 'time': self.state.t}
            step_log = self.step(step_number)
            to_log.update(step_log)
            self.logger.log(to_log)
            self.state.update(self.state.particles, step_size)
            if jnp.isnan(self.state.particles).any():
                raise ValueError(f"Instability detected at step {step_number}")

        return self.state.particles

    def step(self):
        """Take one step, e.g. predict score and move particles. Return a dictionary of values to log."""
        raise NotImplementedError("must be implemented by subclasses")


    def step_size(self, step_number, factor=1e-1):
        """Return the step size for the given step number"""
        if self.step_sizes:
            return self.step_sizes[step_number]
        else: # 1/E(|∇ log π(x)|^2)
            return factor * self.state.particles.shape[0] / jnp.sum(jnp.square(self.target_score(self.state.t, self.state.particles)))


class SDESampler(Sampler):
    """Langevin dynamics sampler"""

    def __init__(self, particles, target_score, step_sizes, max_steps, logger, seed=0, **kwargs):
        super().__init__(particles, target_score, step_sizes, max_steps, logger, **kwargs)
        self.key = jrandom.key(seed)
        self.logger.log_hyperparameters({'seed': seed})

    def step(self, step_number):
        """Equation (4) in https://arxiv.org/pdf/1907.05600"""
        self.key, subkey = jrandom.split(self.key)
        n, dim = self.state.particles.shape
        noise = jrandom.multivariate_normal(subkey, jnp.zeros((n, dim)), jnp.eye(dim))
        t = self.state.t
        drift = self.target_score(t, self.state.particles)
        velocity = self.step_size(step_number) * drift + jnp.sqrt(2 * self.step_size(step_number)) * noise
        self.state.particles += velocity
        return {'noise': noise, 'velocity': velocity}


class ODESampler(Sampler):
    """Deterministic sampler"""


class GDStoppingCriterion:
    """Gradient descent stopping criterion"""
    
    def __call__(self, loss_values, batch_loss_values):
        raise NotImplementedError("must be implemented by subclasses")
    
class FixedNumBatches(GDStoppingCriterion):
    def __init__(self, num_batches=10):
        self.num_batches = num_batches

    def __call__(self, loss_values, batch_loss_values):
        return len(batch_loss_values) >= self.num_batches

class AbsoluteLossChange(GDStoppingCriterion):
    """Stop when the absolute change in loss is below a threshold"""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def __call__(self, loss_values, batch_loss_values):
        if len(loss_values) < 2:
            return False
        return abs(loss_values[-2] - loss_values[-1]) < self.threshold

#TODO: scheduled gd stopping criterion: the number of gd steps is a function of the step number or time

class SBTMSampler(ODESampler):
    """Use a NN to approximate the score"""

    score_model: nnx.Module  # a model to approximate grad-log-density, e.g. a NN
    loss: callable  # loss function to minimize at each step
    optimizer: optax.GradientTransformation  # optimizer for training the model
    mini_batch_size: int  # size of mini-batches
    gd_stopping_criterion: GDStoppingCriterion  # stopping criterion for gradient descent
    debug: bool  # whether to print debug information
    heun: bool  # whether to use Heun's method

    def __init__(self, particles, target_score, step_sizes, max_steps, logger, score_model, loss, optimizer, gd_stopping_criterion=AbsoluteLossChange(), mini_batch_size=100, debug=False, heun=False):
        super().__init__(particles, target_score, step_sizes, max_steps, logger)
        self.score_model = score_model
        self.loss = loss
        self.optimizer = optimizer
        self.gd_stopping_criterion = gd_stopping_criterion
        self.mini_batch_size = mini_batch_size
        self.debug = debug
        self.heun = heun
        self.logger.log_hyperparameters({'mini_batch_size': mini_batch_size, 'heun': heun})
        # self.logger.log_hyperparameters({'optimizer': optimizer, 'gd_stopping_criterion': gd_stopping_criterion})

    def step(self, step_number):
        """Lines 4,5 of algorithm 1 in https://arxiv.org/pdf/2206.04642"""
        loss_values, batch_loss_values = self.train_model()
        score = self.score_model(self.state.particles)
        t = self.state.t
        if self.heun:
            drift = self.target_score(t, self.state.particles) - score
            intermediate_particles = self.state.particles + self.step_size(step_number) * drift
            intermediate_score = self.score_model(intermediate_particles)
            intermediate_drift = self.target_score(t, intermediate_particles) - intermediate_score
            velocity = self.step_size(step_number) * 0.5 * (drift + intermediate_drift)
        else:
            drift = self.target_score(t, self.state.particles) - score
            velocity = self.step_size(step_number) * drift
        self.state.particles += velocity
        if self.debug:
            print(f"Step {step_number}: mean score={jnp.mean(jnp.abs(score)):.3f}, mean velocity={jnp.mean(jnp.abs(velocity)):.3f}, mean particles={jnp.mean(jnp.abs(self.state.particles)):.3f}")
        return {'score': score, 'velocity': velocity, 'loss_values': loss_values, 'batch_loss_values': batch_loss_values}

    def train_model(self):
        """Train the score model using mini-batches"""
        batch_loss_values = []
        loss_values = []
        num_particles = self.state.particles.shape[0]
        num_batches = num_particles // self.mini_batch_size

        while not self.gd_stopping_criterion(loss_values, batch_loss_values):
            loss_values.append(self.loss(self.score_model, self.state.particles))
            for i in range(num_batches):
                batch_start = i * self.mini_batch_size
                batch_end = batch_start + self.mini_batch_size
                batch = self.state.particles[batch_start:batch_end, :]
                loss_value = opt_step(self.score_model, self.optimizer, self.loss, batch)
                batch_loss_values.append(loss_value)
            if self.debug:
                print(f"Loss: {loss_values[-1]}")
        
        loss_values.append(self.loss(self.score_model, self.state.particles))
        return loss_values, batch_loss_values

@nnx.jit(static_argnames='loss')
def opt_step(model, optimizer, loss, batch):
    """Perform one step of optimization"""
    loss_value, grads = nnx.value_and_grad(loss)(model, batch)
    optimizer.update(grads)
    return loss_value
    

class SVGDSampler(ODESampler):
    """Use RKHS to approximate the velocity (target_score - score)"""

    kernel_obj: eqx.Module
    kernel_width: float

    def __init__(self, particles, target_score, step_sizes, max_steps, logger, kernel_obj, kernel_width=-1.):
        """If kernel_width is not passed, it is recomputed at each step"""
        super().__init__(particles, target_score, step_sizes, max_steps, logger)
        self.kernel_obj = kernel_obj
        self.kernel_width = kernel_width
        self.logger.log_hyperparameters({'kernel_obj': kernel_obj, 'kernel_width': kernel_width})

    def step(self, step_number):
        width = self.compute_kernel_width(self.state.particles)
        gradient = self.gradient(width)
        velocity = self.step_size(step_number) * gradient
        self.state.particles += velocity
        return {'kernel_width': float(width), 'velocity': velocity}
    
    @eqx.filter_jit
    def gradient(self, width):
        """Equation (8) in https://arxiv.org/pdf/1608.04471"""
        particles = self.state.particles

        num_particles = particles.shape[0]
        gram_matrix = self.kernel_obj(particles, particles, width)  # k(xⱼ, xᵢ)
        kernel_gradient = self.kernel_obj.gradient_wrt_first_arg(particles, particles, width)  # ∇ⱼ log k(xⱼ, xᵢ)
        score_matrix = self.target_score(self.state.t, particles)  # ∇ log π(x)

        kernel_score_term = jnp.einsum("ij,jk->ik", gram_matrix, score_matrix)  # ∑ⱼ k(xⱼ, xᵢ) ⋅ ∇ log π(xⱼ)
        kernel_gradient_term = jnp.sum(kernel_gradient, axis=0)  # ∑ⱼ ∇ⱼ log k(xⱼ, xᵢ)
        gradient = (kernel_score_term + kernel_gradient_term) / num_particles  # φ(xᵢ) = (1/n) ∑ⱼ k(xⱼ, xᵢ) ⋅ ∇ log π(xⱼ) + ∇ⱼ log k(xⱼ, xᵢ)
        
        return gradient

    @eqx.filter_jit
    def compute_kernel_width(self, particles):
        """Page 6 of https://arxiv.org/pdf/1608.04471 specifies width = med²/log n """
        if self.kernel_width > 0:
            width = self.kernel_width
        else:
            num_particles = particles.shape[0]
            pairwise_sq_distances = jax.vmap(lambda x1: jax.vmap(lambda y1: jnp.sum((x1 - y1) ** 2))(particles))(particles)
            median_distance_sq = jnp.median(pairwise_sq_distances)
            width = median_distance_sq / jnp.log(num_particles)
        return width