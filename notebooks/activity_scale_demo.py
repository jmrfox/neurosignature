# %% [markdown]
# # CTRNN Activity Scale Demo
#
# Demonstrates the three operating regimes from design_2.3.md:
# - **Passive Mode**: Leak-dominated, stable filtering (g=0, r_prop small)
# - **Weakly Active**: Moderate recurrence, richer dynamics (g=0.3, r_prop medium)
# - **Active Mode**: Strong recurrence, nonlinear amplification (g=0.6, r_prop large)

# %%
import numpy as np
import matplotlib.pyplot as plt
from neurosignature.systems import SystemGenerator
from neurosignature.simulation import Simulator

import seaborn as sns

sns.set_context("notebook")

# Disable scientific notation on y-axes
plt.rcParams["axes.formatter.useoffset"] = False

# %% [markdown]
# ## Generate Systems in Different Regimes

# %%
generator = SystemGenerator(n_hidden=32, n_inputs=5, n_outputs=8)

# Get the three regime presets
regimes = generator.generate_regime_comparison(polarity="bipolar")

print("Generated systems:")
for name, system in regimes.items():
    print(
        f"  {name:12s}: g={system.g:.1f}, r_int={system.r_int:.1f}, "
        f"r_prop={system.r_prop:.1f}"
    )

# %% [markdown]
# ## Simulate Same Input Through All Regimes

# %%
# Create input signal
duration_ms = 1000
dt_ms = 1.0
n_steps = int(duration_ms / dt_ms)
time = np.arange(n_steps) * dt_ms

# Random input with bursts
rng = np.random.default_rng(42)
inputs = rng.standard_normal((n_steps, 5)) * 0.2
# Add some stronger bursts
for t in [200, 500, 800]:
    inputs[t : t + 50] += 0.5

# Simulate all three systems
outputs = {}
for name, system in regimes.items():
    simulator = Simulator(system, dt_ms=dt_ms)
    out, _ = simulator.run(inputs)
    outputs[name] = out

# %% [markdown]
# ### Plot Comparison

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

colors = {"passive": "C0", "weakly_active": "C1", "active": "C2"}

for idx, (name, ax) in enumerate(zip(regimes.keys(), axes)):
    out = outputs[name]
    sys = regimes[name]

    # Plot first 4 outputs
    for i in range(min(4, out.shape[1])):
        ax.plot(time, out[:, i], label=f"Out {i}", alpha=0.7)

    ax.axhline(sys.V_rest, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(
        f'{name.replace("_", " ").title()} Mode: '
        f"g={sys.g:.1f}, r_prop={sys.r_prop:.1f}"
    )
    if idx == 0:
        ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (ms)")
plt.suptitle("Operating Regimes Comparison (Same Input)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Activity Scale Sweep
#
# Gradually increase activity level from passive to weakly active.

# %%
# Generate sweep
n_systems = 5
sweep = generator.generate_activity_scale_sweep(n_systems=n_systems, polarity="bipolar")

print(f"Activity scale sweep ({n_systems} systems):")
for i, sys in enumerate(sweep):
    print(f"  System {i}: g={sys.g:.2f}, r_prop={sys.r_prop:.1f}")

# Simulate
sweep_outputs = []
for system in sweep:
    simulator = Simulator(system, dt_ms=dt_ms)
    out, _ = simulator.run(inputs)
    sweep_outputs.append(out)

# %%
# Plot sweep
fig, axes = plt.subplots(n_systems, 1, figsize=(12, 2 * n_systems), sharex=True)

for i, (ax, system, out) in enumerate(zip(axes, sweep, sweep_outputs)):
    for j in range(min(3, out.shape[1])):
        ax.plot(time, out[:, j], alpha=0.7)
    ax.axhline(system.V_rest, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(f"g={system.g:.2f}")
    ax.set_title(f"Activity Level {i+1}/{n_systems} (r_prop={system.r_prop:.1f})")

axes[-1].set_xlabel("Time (ms)")
plt.suptitle("Activity Scale Sweep: Passive → Weakly Active", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Statistics Comparison

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Output variance by regime
regime_names = list(regimes.keys())
regime_vars = [np.var(outputs[name]) for name in regime_names]

ax = axes[0]
bars = ax.bar(range(len(regime_names)), regime_vars, color=["C0", "C1", "C2"])
ax.set_xticks(range(len(regime_names)))
ax.set_xticklabels([n.replace("_", "\n") for n in regime_names])
ax.set_ylabel("Output Variance")
ax.set_title("Activity Level by Regime")

# Sweep variance
sweep_vars = [np.var(out) for out in sweep_outputs]
ax = axes[1]
ax.plot(range(n_systems), sweep_vars, "o-", color="C3")
ax.set_xlabel("System Index")
ax.set_ylabel("Output Variance")
ax.set_title("Variance vs Activity Scale")

# Mean absolute deviation from rest
rest_devs = {}
for name, out in outputs.items():
    sys = regimes[name]
    rest_devs[name] = np.mean(np.abs(out - sys.V_rest))

ax = axes[2]
ax.bar(
    range(len(regime_names)),
    [rest_devs[n] for n in regime_names],
    color=["C0", "C1", "C2"],
)
ax.set_xticks(range(len(regime_names)))
ax.set_xticklabels([n.replace("_", "\n") for n in regime_names])
ax.set_ylabel("Mean |V - V_rest| (mV)")
ax.set_title("Departure from Rest")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Observations
#
# 1. **Passive Mode**: Small perturbations that decay quickly. Output stays near V_rest.
#    Best for: cable equation approximation, passive membrane modeling.
#
# 2. **Weakly Active**: Longer-lasting perturbations, some resonance. Good functional diversity.
#    Best for: exploring system identification, comparing operator families.
#
# 3. **Active Mode**: Large deviations, possible oscillatory behavior, complex dynamics.
#    Best for: testing descriptor robustness, rich temporal structure.
#
# The `generate_activity_scale_sweep()` creates a continuum between these regimes,
# allowing smooth exploration of the dissipation-vs-amplification tradeoff.
