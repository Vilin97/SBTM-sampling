#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.stats import gaussian_kde
from tqdm import trange
jax.config.update("jax_enable_x64", True)

# ----- experiment constants -----
N_PARTICLES = 10_000
DT = 0.002
T_MAX = 2.5
SAVE_TIMES = jnp.arange(0, T_MAX + 1e-9, 1.0)  # 0,1,2,3,4,5
GRID = jnp.linspace(-10.0, 10.0, 1001)  # for KDE & KL
DX = GRID[1] - GRID[0]

# ----- helper functions -----
@jax.jit
def sigma(t):  # variance, not std
    return 1.0 - jnp.exp(-2.0 * (t + 0.1))

@jax.jit
def kl_div(p, q, dx):
    p_safe = jnp.clip(p, 1e-16, None)
    q_safe = jnp.clip(q, 1e-16, None)
    return jnp.sum(p_safe * jnp.log(p_safe / q_safe)) * dx

# Ground‑truth KL between N(0, sigma(t)) and N(0,1)
@jax.jit
def kl_gauss_sigma(s2):
    return 0.5 * (s2 - 1.0 - jnp.log(s2))

# ----- Langevin simulation (Euler–Maruyama) -----
@jax.jit
def langevin_step(x, key):
    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, x.shape)
    x_next = x + (-x) * DT + jnp.sqrt(2.0 * DT) * noise
    return x_next, key

# ----- FP ODE simulation (explicit Euler) -----
@jax.jit
def fp_step(x, t):
    return x + (-1.0 + 1/sigma(t)) * x * DT

#%%
key = jax.random.PRNGKey(1)

# Target density N(0,1) on grid
q_grid = jnp.exp(-0.5 * GRID ** 2) / jnp.sqrt(2 * jnp.pi)

# ----- initial condition -----
key, k0 = jax.random.split(key)
init_std = jnp.sqrt(sigma(0.0))
X0 = init_std * jax.random.normal(k0, (N_PARTICLES,))

# ----- run both systems -----
num_steps = int(T_MAX / DT)
save_indices = (SAVE_TIMES / DT).astype(int)

langevin_samples = []
fp_samples = []

# Store all samples for KL/entropy at every step
all_kl_langevin = []
all_kl_fp = []
all_kl_true = []
all_times = []

def run_simulations_and_kl():
    x_l = X0.copy()
    x_f = X0.copy()
    key_sim = key
    for step in trange(num_steps + 1):
        t = step * DT
        # Save for SAVE_TIMES
        if step in save_indices:
            langevin_samples.append(x_l)
            fp_samples.append(x_f)
        # Compute KL at every step
        kde_l = gaussian_kde(x_l)
        kde_f = gaussian_kde(x_f)
        p_l = kde_l.pdf(GRID)
        p_f = kde_f.pdf(GRID)
        all_kl_langevin.append(kl_div(p_l, q_grid, DX))
        all_kl_fp.append(kl_div(p_f, q_grid, DX))
        all_kl_true.append(kl_gauss_sigma(sigma(t)))
        all_times.append(t)
        # advance one step (skip final extra step after T_MAX)
        if step == num_steps:
            break
        x_l, key_sim = langevin_step(x_l, key_sim)
        x_f = fp_step(x_f, t)

run_simulations_and_kl()

langevin_samples = jnp.stack(langevin_samples)  # (len(SAVE_TIMES), N)
fp_samples = jnp.stack(fp_samples)
all_kl_langevin = jnp.stack(all_kl_langevin)
all_kl_fp = jnp.stack(all_kl_fp)
all_kl_true = jnp.stack(all_kl_true)
all_times = jnp.array(all_times)

#%%
# ----- KDE & KL -----
kl_langevin = []
kl_fp = []
kl_true = []

for i, t in enumerate(SAVE_TIMES):
    kde_l = gaussian_kde(langevin_samples[i])
    kde_f = gaussian_kde(fp_samples[i])
    p_l = kde_l.pdf(GRID)
    p_f = kde_f.pdf(GRID)
    kl_langevin.append(kl_div(p_l, q_grid, DX))
    kl_fp.append(kl_div(p_f, q_grid, DX))
    kl_true.append(kl_gauss_sigma(sigma(t)))

kl_langevin = jnp.stack(kl_langevin)
kl_fp = jnp.stack(kl_fp)
kl_true = jnp.stack(kl_true)

# Numerical −d/dt KL using forward differences (dt=1)
neg_dkl_dt = lambda kl: -(kl[1:] - kl[:-1])  # length 5 (midpoints)
mid_times = (SAVE_TIMES[:-1] + SAVE_TIMES[1:]) / 2

# Numerical −d/dt KL at every step (dt=DT)
neg_dkl_dt_fine = lambda kl: -(kl[1:] - kl[:-1]) / DT
fine_times = (all_times[:-1] + all_times[1:]) / 2

#%%
# ----- visualization -----
plt.figure(figsize=(12, 4))
for i, t in enumerate(SAVE_TIMES):
    plt.subplot(2, 3, i + 1)
    xs = GRID
    kde_l = gaussian_kde(langevin_samples[i])
    kde_f = gaussian_kde(fp_samples[i])
    plt.plot(xs, kde_l.pdf(xs), label="Langevin", lw=1)
    plt.plot(xs, kde_f.pdf(xs), label="FP ODE", lw=1)
    std = jnp.sqrt(sigma(t))
    plt.plot(xs, jnp.exp(-0.5 * (xs / std) ** 2) / (std * jnp.sqrt(2 * jnp.pi)), label="True", lw=1)
    plt.title(f"t={int(t)}")
    plt.xlim(-4, 4)
    if i == 0:
        plt.legend(fontsize=7)
plt.tight_layout()
plt.show()

#%%
# KL divergence plot at every time step
plt.figure()
plt.plot(all_times, all_kl_langevin, label="Langevin")
plt.plot(all_times, all_kl_fp, label="FP ODE")
plt.plot(all_times, all_kl_true, label="True Gaussian")
plt.xlabel("t")
plt.ylabel("KL(N(0,σ^2(t)) || N(0,1))")
plt.yscale("log")
plt.legend(); plt.grid(True)
plt.title("KL divergence vs time (all steps)")
plt.show()

# −d/dt KL plot at every time step

plt.figure()
plt.plot(fine_times, neg_dkl_dt_fine(all_kl_langevin), label="−d/dt KL (Langevin)")
plt.plot(fine_times, neg_dkl_dt_fine(all_kl_fp), label="−d/dt KL (FP ODE)")
plt.plot(fine_times, neg_dkl_dt_fine(all_kl_true), label="−d/dt KL (True)")
plt.xlabel("t")
plt.ylabel("−d/dt KL")
plt.yscale('symlog', linthresh=1e-6)
plt.legend(); plt.grid(True)
plt.title("KL decay rate vs time (all steps)")
plt.show()

# %%
