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

# Classes 
class Sampler:
    particles # jnp array of particles
    target_score # target score function
    step_size # delta t
    max_steps # maximum number of steps
    logger # logger object

    def sample(self):
        """Sample from the target distribution"""
        for _ in range(self.max_steps):
            self.step()
            logger.log(...)

        return self.particles

    def step(self):
        """Take one step, e.g. predict score and move particles"""
        raise NotImplementedError("must be implemented by subclasses")


class SDESampler(Sampler):
    """Langevin dynamics sampler"""
    
    def step(self):
        """Draw brownian noise Z and move particles in direction target_score - sqrt(2)Z"""        
        pass


class ODESampler(Sampler):
    """Deterministic sampler"""
    score_model # a model to approximate grad-log-density, e.g. a NN

    def step(self):
        """Train the score model and move particles in direction target_score - score_model(particles)"""
        pass


class SBTMSampler(ODESampler):
    """Use a NN to approximate the score"""
    # assume that the score model has already been pre-trained

class SVGDSampler(ODESampler):
    """Use RKHS to approximate the score"""

# ScoreModel class to approximate the score function. It should have methods `fit` and `predict`. `predict` is the forward pass of the model. `fit` should fit the model given particle locations. The class is inherited by NNScoreModel and RKHSScoreModel. NNScoreModel should contain a Flax model, a loss, an optimizer, and a callable `stop_gd` that stops the gradient descent (it can perform a fixed number of GD steps or implement a more sophisticated stopping criterion). RKHSScoreModel should contain the kernel function.

# Logger class to log target score, step_size, max-steps, step number, particle locations, predicted score, score model. Should have method `log`.

# Experiment class to orchestrate running a single experiment. It should take a `config` dict. It should have methods `run` and `compute_metrics`. `run` should load the model (or train a new one if not available), init the sampler, run the sampler, and save the config and the results. `compute_metrics` should load the results, compute metrics and save the metrics. `plot_metrics` should load the metrics, plot them, and save the plots. 

# DataManager class to save and load data. Should save and load experiment data and plots. Not sure how to imlpement this yet. Maybe with pickle or maybe with a PostgreSQL database or maybe with pandas.

