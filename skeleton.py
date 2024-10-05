"""Project skeleton"""

"""
What do I want from the code?
- compare different samplers: SBTM, SVGD, Langevin dynamics
- compare different NN architectures for SBTM: MLP, ResNet, transformer
- compare different numbers of SGD steps for SBTM
- compare different loss functions for SBTM
- compute and plot various metrics: Fisher divergence, KL divergence, moments, score approximation error, etc.
- save results and intermediate states and run analysis afterwards
- pre-train initial NNs and save them
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

    def sample(self):
        """Sample from the target distribution"""
        for _ in range(self.max_steps):
            self.step()
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

class SVGDSampler(ODESampler):
    """Use RKHS to approximate the score"""