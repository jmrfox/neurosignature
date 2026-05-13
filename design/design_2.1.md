# CTRNN Update Design: Subthreshold Resting-Potential Dynamics

## 1. Motivation

The original CTRNN implementation behaves like:

- A generic recurrent dynamical system
- Centered around zero
- Potentially capable of self-sustained activity

However, biological subthreshold membrane dynamics behave differently.

Desired properties:

- Membrane voltage fluctuates around a resting potential
- Synaptic input produces transient perturbations
- Activity naturally decays back to baseline
- No persistent autonomous activity
- Smooth low-frequency voltage traces
- Support for excitation-only, inhibition-only, and bipolar modes

The updated CTRNN should therefore model **stable dissipative filtering dynamics** rather than autonomous recurrent activity.

---

## 2. Core Conceptual Change

Instead of interpreting hidden states as **arbitrary neural activations**, interpret them as **deviations from resting equilibrium**.

The system should satisfy:

**If input $u(t) = 0$:**

$$h(t) \rightarrow 0$$

**Therefore:**

$$v(t) \rightarrow V_{rest}$$

---

## 3. Updated Dynamical System

### Hidden State Dynamics

$$
\frac{dh}{dt} = \frac{-h + W_h \phi(h) + W_u u(t)}{\tau}
$$

**Where:**

| Symbol | Description |
|--------|-------------|
| $h$ | Hidden state vector |
| $W_h$ | Recurrent connectivity matrix |
| $\phi$ | Smooth bounded nonlinearity |
| $W_u$ | Input projection matrix |
| $\tau$ | Leak timescale(s) |

The critical term **$-h$** acts as:

- Membrane leak
- Passive relaxation
- Dissipative stabilization

---

## 4. Output Definition

Outputs are now interpreted as **voltage perturbations around a resting potential**.

$$
v(t) = V_{rest} + \alpha W_o h(t)
$$

**Where:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| $V_{rest}$ | Resting membrane potential | -65 mV |
| $\alpha$ | Voltage scaling factor | Variable |
| $W_o$ | Output projection matrix | Learned/random |

---

## 5. Stability Requirements

To maintain biologically realistic subthreshold behavior:

- Recurrent gain must remain weak
- Leak must dominate long-term dynamics

**Recommended:**

$$\rho(W_h) < 0.7$$

**Suggested range:** $0.2 - 0.6$

This ensures:

- Bounded responses
- Transient perturbations
- Return to baseline
- Smooth voltage traces

**Avoid:**

- Oscillatory attractors
- Chaos
- Self-sustained activity

*For initial development.*

---

## 6. Polarity Modes

The simulator should support three operating modes:

### 6.1 Excitatory-Only Mode

**Behavior:**

- Voltage deviations are positive only
- Membrane depolarizes away from baseline
- All activity remains above $V_{rest}$

**Output:**

$$
v(t) = V_{rest} + \text{softplus}(W_o h(t))
$$

**Alternative:** ReLU

**Constraints:**

- $W_u \geq 0$
- Optionally $W_o \geq 0$

**Result:** Positive-going EPSP-like fluctuations

---

### 6.2 Inhibitory-Only Mode

**Behavior:**

- Voltage deviations are negative only
- Membrane hyperpolarizes below baseline

**Output:**

$$
v(t) = V_{rest} - \text{softplus}(W_o h(t))
$$

**Result:** IPSP-like fluctuations

---

### 6.3 Bipolar Mode

**Behavior:**

- Both depolarization and hyperpolarization allowed
- Closest to realistic mixed E/I subthreshold activity

**Output:**

$$
v(t) = V_{rest} + W_o h(t)
$$

**Constraints:** None (no polarity constraints)

---

## 7. Synaptic Input Processing

Inputs remain generated from:

- Master Poisson process
- Routing distribution $P$

Each synaptic event is filtered using a smooth synaptic kernel.

### 7.1 Synaptic Kernel

Recommended alpha-function kernel:

$$
\alpha(t) = H(t) \cdot \frac{t}{\tau_s} \cdot \exp\left(-\frac{t}{\tau_s}\right)
$$

**Where:**

| Symbol | Description |
|--------|-------------|
| $H(t)$ | Heaviside step function |
| $\tau_s$ | Synaptic timescale |

**Produces:**

- Smooth EPSP-like currents
- Realistic temporal filtering

---

## 8. Heterogeneous Timescales

To increase dynamical richness, use **per-unit timescales** $\tau_i$ rather than a single scalar $\tau$.

**Suggested sampling:**

$$
\tau_i \sim \text{LogUniform}(10\text{ ms}, 200\text{ ms})
$$

**Creates:**

- Multiple decay modes
- Richer autocorrelation structure
- Compartment-like temporal diversity

---

## 9. Recommended Discrete-Time Update

**Euler discretization:**

$$
h[t+1] = h[t] + \frac{dt}{\tau} \cdot \left(-h[t] + W_h \tanh(h[t]) + W_u u[t]\right)
$$

**Outputs:**

$$
v[t] = V_{rest} + \alpha W_o h[t]
$$

Apply polarity transform after projection.

---

## 10. Suggested Initial Parameters

| Category | Parameter | Value |
|----------|-----------|-------|
| **Simulation** | $dt$ | 1 ms |
| | $T$ | 10 seconds |
| **Hidden System** | $N_H$ | 64 |
| | Sparsity | 0.1 |
| | Spectral radius | 0.5 |
| **Voltage** | $V_{rest}$ | -65 mV |
| | $\alpha$ | 5 mV |
| **Synaptic** | $\tau_s$ | 10 ms |
| **Timescales** | $\tau_i$ | LogUniform(10 ms, 100 ms) |

---

## 11. Expected Qualitative Behavior

The updated system should produce:

- Smooth voltage fluctuations
- Bounded dynamics
- Transient depolarizations
- Relaxation to baseline
- No persistent activity
- Low-frequency structure
- Morphology-like filtering behavior

---

## 12. Interpretation

The updated CTRNN should now be interpreted as a **generalized nonlinear passive membrane surrogate** rather than a generic recurrent neural network.

The hidden dynamics approximate:

$$
C \frac{dv}{dt} = -g_L(v - V_{rest}) + I_{syn} + I_{coupling}
$$

This creates a conceptual bridge toward:

- Future cable models
- Graph diffusion systems
- Arbor simulations

While remaining:

- Lightweight
- Scalable
- Easy to analyze

---

## 13. Development Priorities

### Implementation Order

1. Resting-potential outputs
2. Leak-dominated hidden dynamics
3. Polarity modes
4. Heterogeneous timescales
5. Stable recurrent initialization
6. Synaptic alpha kernels
7. Descriptor/statistics updates

### Future Additions (After Validation)

- [ ] Spikes
- [ ] Inhibition/excitation mixtures
- [ ] Plasticity
- [ ] Morphology dependence
- [ ] Biophysical mechanisms
