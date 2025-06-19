#%%
import os, sys, functools
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"]       = "off"

import jax, jax.numpy as jnp, jax.random as jr
from flax import nnx
import optax, matplotlib.pyplot as plt
from tqdm import trange

# ---------- data: 2-component Gaussian mixture -----------------------
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sbtm import distribution                                   # your module
f_dist  = distribution.GaussianMixture(
            [jnp.array([-2.5,0]), jnp.array([ 2.5,0])],
            [jnp.eye(2),          jnp.eye(2)          ],
            [0.5, 0.5])
f_pdf   = f_dist.density
f_score = f_dist.score

KEY          = jr.PRNGKey(0)
KEY, k_data  = jr.split(KEY)
X_ALL        = f_dist.sample(k_data, 2**12)                    # ~4 k pts

xx, yy       = jnp.meshgrid(jnp.linspace(-6,6,200),
                            jnp.linspace(-3,3,200))
GRID         = jnp.stack([xx.ravel(), yy.ravel()], -1)
F_GRID       = jnp.reshape(f_pdf(GRID), xx.shape)

rnd_idx      = jr.choice(KEY, X_ALL.shape[0], (20,), replace=False)

# ---------- common model ---------------------------------------------
class CondMLP(nnx.Module):
    def __init__(self, d_x, h=[128,128,128], act=nnx.soft_sign, seed=0):
        rngs, inp = nnx.Rngs(seed), d_x+1
        # First hidden layer: input size is d_x+1
        self.hidden = [nnx.Linear(inp, h[0], rngs=rngs)]
        # Subsequent hidden layers: input size is previous hidden size
        for i in range(1, len(h)):
            self.hidden.append(nnx.Linear(h[i-1], h[i], rngs=rngs))
        self.out    = nnx.Linear(h[-1], d_x, rngs=rngs)
        self.act    = act
    def __call__(self, x, sigma):
        sigma = jnp.asarray(sigma, x.dtype).reshape(-1,1)
        sigma = jnp.broadcast_to(sigma, (x.shape[0],1))
        h     = jnp.concatenate([x, sigma], -1)
        for lyr in self.hidden: h = self.act(lyr(h))
        return self.out(h)

# ---------- KDE utilities --------------------------------------------
def median_bandwidth(x):
    d = jnp.median(jnp.linalg.norm(x[None]-x[:,None], axis=-1))
    return 1.06 * d * (x.shape[0] ** (-1/5))

def kde_score_fn(x_eval, x_train, h):
    """∇log p_KDE(x_eval) for Gaussian KDE (isotropic, bandwidth h)."""
    def score_at_point(xq):
        dif = (xq - x_train) / h                       # (N,2)
        w   = jnp.exp(-0.5*jnp.sum(dif**2,1))
        num = jnp.sum(-dif * w[:,None], 0)
        den = jnp.sum(w)
        return num / (h**2 * den)
    return jax.vmap(score_at_point)(x_eval)

# ---------- loss functions (Hyvärinen & SSM use Hutchinson) -----------
@nnx.jit
def dsm_loss(model, x, rng, σ_min=0.1, σ_max=2.0):
    kσ,kε  = jr.split(rng)
    σ      = jr.uniform(kσ, (x.shape[0],1), dtype=x.dtype, minval=σ_min, maxval=σ_max)
    ε      = jr.normal(kε,x.shape)
    x_noisy= x + σ*ε
    pred   = model(x_noisy, σ)
    return 0.5*jnp.mean(jnp.sum((pred + ε/σ)**2,-1) * σ**2)

def _hutch_trace(fn, x, v):                               # Tr[∇fn] via Hutchinson
    _, hvp = jax.jvp(fn,   (x,), (v,))
    return jnp.sum(v * hvp)

def hyvarinen_loss(model, x, rng):
    v = jr.normal(rng, x.shape)                           # (B,2)
    def per_row(xi, vi):
        g    = model(xi[None], 0.0)[0]
        trace= _hutch_trace(lambda y: model(y[None],0.0)[0], xi, vi)
        return 0.5*jnp.sum(g**2) + trace
    return jnp.mean(jax.vmap(per_row)(x, v))

def ssm_loss(model, x, rng, n_proj=4):
    k_v, _ = jr.split(rng)
    v = jr.normal(k_v,(n_proj,2))
    v = v / jnp.linalg.norm(v,axis=1,keepdims=True)
    def proj_term(vv):
        proj_score = jnp.sum(model(x,0.0) * vv, 1)        # scalar per sample
        div_est    = jax.vmap(lambda xi:
                        _hutch_trace(lambda y: model(y[None],0.0)[0], xi, vv))(x)
        return 0.5*jnp.mean(proj_score**2 + div_est)
    return jnp.mean(jax.vmap(proj_term)(v))

# ---------- training wrapper -----------------------------------------
TRAIN_STEPS = 100
BATCH       = 1024
LR          = 1e-3
EMA_ALPHA   = 0.95
SIGMA_EVAL  = 0.5

def ema(arr, a=EMA_ALPHA):
    out, m = [], arr[0]
    for v in arr: m = a*m + (1-a)*v; out.append(m)
    return jnp.array(out)

@nnx.jit(static_argnames=("loss_fn",))
def train_step(model,opt,x_batch,rng,loss_fn):
    rng, kr = jr.split(rng)
    l, gr   = nnx.value_and_grad(loss_fn)(model,x_batch,kr)
    opt.update(gr)
    return l, rng

def train_method(name, loss_fn):
    key, mdl = jr.PRNGKey(42), CondMLP(2)
    opt      = nnx.Optimizer(mdl,
                  optax.chain(optax.clip_by_global_norm(0.01),
                              optax.adamw(LR)))
    losses, explicit = [], []
    rng = key
    for step in trange(TRAIN_STEPS,desc=f"Training {name}",leave=False):
        rng, k_sample = jr.split(rng)
        idx  = jr.choice(k_sample, X_ALL.shape[0], (BATCH,), replace=False)
        l, rng = train_step(mdl,opt,X_ALL[idx],rng,loss_fn)
        losses.append(l)
        if step % 10 == 0:
            explicit.append(jnp.mean(
                (mdl(X_ALL, SIGMA_EVAL) - f_score(X_ALL))**2))
    return mdl, jnp.array(losses), jnp.array(explicit)

#%%
# ---------- run all four methods --------------------------------------
METHODS = {
    "KDE":       {"train": False},
    "DSM":       {"train": True,  "loss": dsm_loss},
    "Hyvarinen": {"train": True,  "loss": hyvarinen_loss},
    "SSM":       {"train": True,  "loss": functools.partial(ssm_loss,n_proj=4)},
}

results = {}

for name, cfg in METHODS.items():
    if cfg["train"]:
        mdl, prim, expl = train_method(name, cfg["loss"])
        results[name] = dict(model=mdl, prim=prim, expl=expl)
    else:                                                # KDE
        h   = median_bandwidth(X_ALL)
        def kde_model(xq, sigma):   # dummy sigma arg for API parity
            return kde_score_fn(xq, X_ALL, h)
        mse = jnp.mean((kde_model(X_ALL,0)-f_score(X_ALL))**2)
        results[name] = dict(model=kde_model,
                             prim=jnp.array([mse]),
                             expl=jnp.array([mse]))

#%%
# ---------- plotting ---------------------------------------------------
for name, res in results.items():
    prim, expl = res["prim"], res["expl"]
    # 1) training-loss (skip for KDE)
    if name != "KDE":
        plt.figure(figsize=(8,1.5))
        plt.plot(prim, label=f"{name} loss")
        if prim.size > 1: plt.plot(ema(prim), label="EMA")
        plt.legend(); plt.title(f"{name} – primary loss"); plt.show()
    # 2) explicit MSE to true score
    if name != "KDE":
        xs = jnp.arange(0,len(expl))*10 if expl.size>1 else jnp.array([0])
        plt.figure(figsize=(8,1.5))
        plt.plot(xs, expl, label="MSE σ=0.5")
        plt.title(f"{name} – explicit MSE"); plt.legend(); plt.show()
    # 3) quiver of learned vs true scores
    mdl = res["model"]
    plt.figure(figsize=(8,4))
    plt.contourf(xx,yy,F_GRID,30,alpha=.35,cmap='Greens')
    plt.scatter(X_ALL[:,0], X_ALL[:,1], s=8, c='k')
    for i,idx in enumerate(rnd_idx):
        p   = X_ALL[idx]
        tru = -f_score(p)
        est = -mdl(p[None],0.1)[0]
        if i==0:
            plt.arrow(p[0], p[1], tru[0], tru[1], width=.01, head_width=.07, color='r', label='-true')
            plt.arrow(p[0], p[1], est[0], est[1], width=.01, head_width=.07, color='b', label='-est')
        else:
            plt.arrow(p[0], p[1], tru[0], tru[1], width=.01, head_width=.07, color='r')
            plt.arrow(p[0], p[1], est[0], est[1], width=.01, head_width=.07, color='b')
        plt.scatter(*p,c='y',s=50,edgecolors='k')
    # Add MSE loss as text on the plot
    mse_val = float(jnp.mean((mdl(X_ALL, 0.1) - f_score(X_ALL))**2))
    plt.text(0.98, 0.02, f"MSE: {mse_val:.4f}", fontsize=12, color='k',
             ha='right', va='bottom', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.legend(); plt.axis('off')
    plt.title(f"{name} – learned vs true score"); plt.show()

# %%
