# %% [markdown]
# # ContinuousTimeRNN Exploration: Parameter Effects on Trace Variability
#
# This notebook explores how different parameters affect the dynamics and trace
# variability of `ContinuousTimeRNN` systems according to the design_2.2.md
# specification.
#
# Key parameters:
# - **r_int**: Integration matrix spectral radius (0.3-0.8)
# - **r_prop**: Propagation matrix spectral radius (0.05-0.3)
# - **g**: Global recurrent gain (0.0-0.3)
# - **sparsity**: Fraction of non-zero recurrent connections
# - **tau**: Timescale(s) for hidden state decay
# - **polarity**: bipolar, excitatory-only, or inhibitory-only
# - **V_rest**: Resting membrane potential (default -65 mV)

# %%
import numpy as np
import matplotlib.pyplot as plt
from neurosignature.systems import ContinuousTimeRNN, SystemGenerator
from neurosignature.simulation import Simulator

import seaborn as sns
from matplotlib.ticker import ScalarFormatter

sns.set_context("notebook")

# Disable scientific notation on y-axes
plt.rcParams["axes.formatter.useoffset"] = False

# %% [markdown]
# ## 1. Basic System Setup and Single Trace
#
# Create a passive subthreshold CTRNN and visualize its output traces.

# %%
# System parameters
n_hidden = 32
n_inputs = 5
n_outputs = 8
duration_ms = 1000
dt_ms = 1.0
n_steps = int(duration_ms / dt_ms)

# Create the CTRNN with new API
system = ContinuousTimeRNN(
    n_hidden=n_hidden,
    n_inputs=n_inputs,
    n_outputs=n_outputs,
    r_int=0.5,  # Integration matrix radius
    r_prop=0.1,  # Propagation matrix radius
    g=0.1,  # Global gain
    V_rest=-65.0,  # Resting potential
    polarity="bipolar",
    sparsity=0.1,
    seed=42,
)

# Generate random input
rng = np.random.default_rng(42)
inputs = rng.standard_normal((n_steps, n_inputs)) * 0.3

# Simulate
simulator = Simulator(system, dt_ms=dt_ms)
outputs, states = simulator.run(inputs, record_states=True)

print(f"Output shape: {outputs.shape}")
print(f"States shape: {states.shape}")
print(f"Tau range: {system.tau.min():.1f} - {system.tau.max():.1f} ms")
print(f"V_rest: {system.V_rest} mV")

# %% [markdown]
# ### Plot output traces

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

time = np.arange(n_steps) * dt_ms

# Plot outputs
ax = axes[0]
for i in range(min(n_outputs, 5)):  # Plot first 5 outputs
    ax.plot(time, outputs[:, i], label=f"Output {i}", alpha=0.8)
ax.axhline(system.V_rest, color="gray", linestyle="--", alpha=0.5, label="V_rest")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.set_title("Output Traces (Voltage)")
ax.legend(loc="upper right", fontsize=8)

# Plot a few hidden units
ax = axes[1]
n_plot = 5
for i in range(n_plot):
    ax.plot(time, states[:, i], label=f"Hidden {i}", alpha=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Hidden State")
ax.set_title(f"Sample Hidden State Traces ({n_plot} of {n_hidden})")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Effect of Integration Matrix (W_int, r_int)
#
# The integration matrix W_int controls latent mixing before the nonlinearity.
# Higher r_int values increase computational capacity but can affect stability.

# %%
# Sweep r_int values
r_int_values = [0.2, 0.5, 0.8]
n_systems = len(r_int_values)

systems = []
for r_int in r_int_values:
    sys = ContinuousTimeRNN(
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        r_int=r_int,
        r_prop=0.1,
        g=0.1,
        polarity="bipolar",
        sparsity=0.1,
        seed=42,
    )
    systems.append(sys)

# Run same input through all systems
simulator = Simulator(systems[0], dt_ms=dt_ms)
all_outputs = simulator.run_with_multiple_systems(systems, inputs)

print(f"All outputs shape: {all_outputs.shape}")

# %%
# Plot traces for different r_int values
fig, axes = plt.subplots(n_systems, 1, figsize=(12, 2 * n_systems), sharex=True)

for i, (ax, r_int) in enumerate(zip(axes, r_int_values)):
    for j in range(min(n_outputs, 3)):
        ax.plot(time, all_outputs[i, :, j], label=f"Out {j}", alpha=0.7)
    ax.axhline(system.V_rest, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(f"r_int={r_int}")
    ax.set_title(f"Integration Matrix Radius = {r_int}")

axes[-1].set_xlabel("Time (ms)")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
plt.suptitle("Effect of Integration Matrix Radius (r_int)", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Effect of Propagation Matrix (r_prop)
#
# The propagation matrix W_prop controls recurrent coupling between latent modes.
# It is symmetric and should remain weak (0.05-0.3) for stable dynamics.

# %%
# Sweep r_prop values
r_prop_values = [0.05, 0.1, 0.2]
n_prop = len(r_prop_values)

prop_systems = []
for r_prop in r_prop_values:
    sys = ContinuousTimeRNN(
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        r_int=0.5,
        r_prop=r_prop,
        g=0.1,
        polarity="bipolar",
        sparsity=0.1,
        seed=42,
    )
    prop_systems.append(sys)

# Run simulation
simulator = Simulator(prop_systems[0], dt_ms=dt_ms)
prop_outputs = simulator.run_with_multiple_systems(prop_systems, inputs)

# %%
# Plot traces for different r_prop values
fig, axes = plt.subplots(n_prop, 1, figsize=(12, 2 * n_prop), sharex=True)

for i, (ax, r_prop) in enumerate(zip(axes, r_prop_values)):
    for j in range(min(n_outputs, 3)):
        ax.plot(time, prop_outputs[i, :, j], label=f"Out {j}", alpha=0.7)
    ax.axhline(system.V_rest, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(f"r_prop={r_prop}")
    ax.set_title(f"Propagation Matrix Radius = {r_prop}")

axes[-1].set_xlabel("Time (ms)")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
plt.suptitle("Effect of Propagation Matrix Radius (r_prop)", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Effect of Global Gain (g)
#
# Global gain g scales the entire recurrent term. Small values (0.0-0.3)
# ensure stable dissipative dynamics without self-sustained activity.

# %%
# Sweep gain values
gain_values = [0.0, 0.1, 0.4]
n_gains = len(gain_values)

gain_systems = []
for g in gain_values:
    sys = ContinuousTimeRNN(
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        r_int=0.5,
        r_prop=0.1,
        g=g,
        polarity="bipolar",
        sparsity=0.1,
        seed=42,
    )
    gain_systems.append(sys)

# Run simulation
simulator = Simulator(gain_systems[0], dt_ms=dt_ms)
gain_outputs = simulator.run_with_multiple_systems(gain_systems, inputs)

# %%
# Plot traces for different gain values
fig, axes = plt.subplots(n_gains, 1, figsize=(12, 2 * n_gains), sharex=True)

for i, (ax, g) in enumerate(zip(axes, gain_values)):
    for j in range(min(n_outputs, 3)):
        ax.plot(time, gain_outputs[i, :, j], label=f"Out {j}", alpha=0.7)
    ax.axhline(system.V_rest, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(f"g={g}")
    ax.set_title(f"Global Gain = {g}")

axes[-1].set_xlabel("Time (ms)")
axes[0].legend(loc="upper right", fontsize=8, ncol=3)
plt.suptitle("Effect of Global Gain (g)", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Polarity Modes Comparison
#
# Compare bipolar, excitatory-only, and inhibitory-only modes.
# - **Bipolar**: Can depolarize and hyperpolarize (most general)
# - **Excitatory**: Only depolarizes above V_rest
# - **Inhibitory**: Only hyperpolarizes below V_rest

# %%
generator = SystemGenerator(n_hidden=n_hidden, n_inputs=n_inputs, n_outputs=n_outputs)

# Generate polarity comparison
polarity_systems = generator.generate_polarity_comparison(
    base_params={"g": 0.1, "r_int": 0.5, "r_prop": 0.1, "sparsity": 0.1}
)

# Run simulations
polarity_outputs = {}
for polarity, sys in polarity_systems.items():
    simulator = Simulator(sys, dt_ms=dt_ms)
    outputs, _ = simulator.run(inputs)
    polarity_outputs[polarity] = outputs

# %%
# Plot polarity comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

colors = {"bipolar": "C0", "excitatory": "C1", "inhibitory": "C2"}

for idx, (polarity, ax) in enumerate(
    zip(["bipolar", "excitatory", "inhibitory"], axes)
):
    outputs = polarity_outputs[polarity]
    for j in range(min(n_outputs, 3)):
        ax.plot(time, outputs[:, j], label=f"Out {j}", alpha=0.7)

    # Show V_rest line
    sys = polarity_systems[polarity]
    ax.axhline(sys.V_rest, color="gray", linestyle="--", alpha=0.5)

    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"{polarity.capitalize()} Mode (V_rest = {sys.V_rest} mV)")

    # Check polarity constraints
    if polarity == "excitatory":
        assert np.all(outputs >= sys.V_rest - 1e-6)
    elif polarity == "inhibitory":
        assert np.all(outputs <= sys.V_rest + 1e-6)

axes[0].legend(loc="upper right", fontsize=8, ncol=3)
axes[-1].set_xlabel("Time (ms)")
plt.suptitle("Polarity Modes Comparison", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Resting Potential Behavior
#
# Verify that the system returns to resting potential when input is zero.

# %%
# Create system
rest_system = ContinuousTimeRNN(
    n_hidden=20,
    n_inputs=5,
    n_outputs=3,
    g=0.1,
    V_rest=-70.0,
    polarity="bipolar",
    seed=42,
)

# Initialize with non-zero state
h = np.random.randn(20) * 0.5
states_over_time = [h.copy()]

# Run with zero input
zero_input = np.zeros(5)
for _ in range(500):
    h = rest_system.step(h, zero_input, dt=1.0)
    states_over_time.append(h.copy())

states_array = np.array(states_over_time)
outputs_over_time = np.array([rest_system.compute_output(h) for h in states_over_time])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Hidden states
ax = axes[0]
for i in range(5):
    ax.plot(states_array[:, i], label=f"Unit {i}", alpha=0.7)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Time Step")
ax.set_ylabel("Hidden State")
ax.set_title("Hidden States Decay to Zero")
ax.legend()

# Outputs
ax = axes[1]
for i in range(3):
    ax.plot(outputs_over_time[:, i], label=f"Output {i}", alpha=0.7)
ax.axhline(rest_system.V_rest, color="red", linestyle="--", alpha=0.7, label="V_rest")
ax.set_xlabel("Time Step")
ax.set_ylabel("Voltage (mV)")
ax.set_title(f"Outputs Decay to V_rest = {rest_system.V_rest} mV")
ax.legend()

plt.tight_layout()
plt.show()

print(f"Final output: {outputs_over_time[-1]}")
print(f"Expected (V_rest): {rest_system.V_rest}")
print(f"Error: {np.abs(outputs_over_time[-1] - rest_system.V_rest).max():.4f} mV")

# %% [markdown]
# ## 7. Summary: Parameter Effects

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# r_int effect
ax = axes[0, 0]
r_int_std = [np.std(all_outputs[i]) for i in range(n_systems)]
ax.plot(r_int_values, r_int_std, "o-", color="C0")
ax.set_xlabel("r_int (Integration Radius)")
ax.set_ylabel("Output Std Dev")
ax.set_title("Effect of Integration Matrix")
ax.grid(True, alpha=0.3)

# r_prop effect
ax = axes[0, 1]
r_prop_std = [np.std(prop_outputs[i]) for i in range(n_prop)]
ax.plot(r_prop_values, r_prop_std, "o-", color="C1")
ax.set_xlabel("r_prop (Propagation Radius)")
ax.set_ylabel("Output Std Dev")
ax.set_title("Effect of Propagation Matrix")
ax.grid(True, alpha=0.3)

# Global gain effect
ax = axes[1, 0]
gain_std = [np.std(gain_outputs[i]) for i in range(n_gains)]
ax.plot(gain_values, gain_std, "o-", color="C2")
ax.set_xlabel("Global Gain (g)")
ax.set_ylabel("Output Std Dev")
ax.set_title("Effect of Global Gain")
ax.grid(True, alpha=0.3)

# Polarity comparison
ax = axes[1, 1]
for polarity in ["bipolar", "excitatory", "inhibitory"]:
    outputs = polarity_outputs[polarity]
    mean_v = np.mean(outputs)
    std_v = np.std(outputs)
    ax.errorbar([polarity], [mean_v], yerr=[std_v], fmt="o", capsize=5, label=polarity)
ax.axhline(-65, color="gray", linestyle="--", alpha=0.5, label="V_rest")
ax.set_ylabel("Mean Voltage (mV)")
ax.set_title("Polarity Modes")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Parameter Effects on Trace Variability - Summary", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# 1. **r_int (Integration)**: Controls computational capacity. Higher values allow
#    more complex latent computations.
#
# 2. **r_prop (Propagation)**: Controls recurrent coupling. Should remain weak
#    (0.05-0.3) for stable dissipative dynamics.
#
# 3. **Global Gain (g)**: Scales overall recurrence. Very small values recommended
#    for passive filtering behavior.
#
# 4. **Polarity Modes**:
#    - Bipolar: Most general, allows both depolarization and hyperpolarization
#    - Excitatory: EPSP-like, only positive deviations from rest
#    - Inhibitory: IPSP-like, only negative deviations from rest
#
# 5. **Resting Potential**: The system naturally decays to V_rest when input is
#    zero, ensuring passive dissipative behavior (no self-sustained activity).
#
# These parameters can be tuned to achieve desired dynamical regimes for
# morphology modeling and system identification tasks.
