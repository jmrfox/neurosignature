# Project Design Document: Functional System Identification of Dynamical Neural Operators

## 1. Project Overview

## Goal

Develop a framework for:

1. Generating synthetic dynamical neural systems
2. Driving them with stochastic synaptic-like inputs
3. Recording multi-channel voltage-like outputs
4. Compressing outputs into fixed-dimensional descriptors
5. Comparing systems using a functional distance metric

The long-term goal is to compare neuronal morphologies, plasticity-induced structural changes, and dynamical computational capability.

However, the initial implementation will use synthetic recurrent dynamical systems, not full biophysical Arbor simulations.

At time of writing, I do not plan to implement full Arbor simulations in this project. The neurosignature framework will be developed here and applied to simulations in a separate project.

## 2. High-Level Mathematical Structure

We define a dynamical system $F_\theta : U \rightarrow V$ where:

- $\theta$ = system parameters
- U = space of input event streams
- V = space of output voltage traces

The pipeline becomes:

```text
Θ
  ↓
Generate dynamical system F_θ
  ↓
Generate stochastic inputs U
  ↓
Simulate outputs V
  ↓
Compute summary statistics W(V)
  ↓
Compute system descriptors S(F_θ)
  ↓
Compute distances D(F_θ1, F_θ2)
```

## 3. Initial Simplifying Assumptions

### Restrict to

- Deterministic systems
- Subthreshold dynamics only
- Excitation-only inputs
- Continuous-valued voltage traces
- Stable bounded dynamics
- No spikes
- No inhibition
- No plasticity during simulation

This avoids threshold nonlinearities, spike chaos, reset discontinuities, and difficult event alignment.

## 4. Core System Model

### Continuous-Time Recurrent Neural Dynamical System

**State:** $h(t) \in \mathbb{R}^{N_H}$

**Inputs:** $u(t) \in \mathbb{R}^{N_S}$

**Outputs:** $v(t) \in \mathbb{R}^{N_C}$

**Dynamics:**

$$\frac{dh}{dt} = \frac{-h + \tanh(W_h h + W_u u + b_h)}{\tau}$$

**Output equation:**

$$v = W_o h + b_o$$

**Where:**

- W_h : recurrent hidden weights
- W_u : input projection weights
- W_o : output projection weights
- τ : hidden state timescale(s)
- tanh : smooth bounded nonlinearity

## 5. System Requirements

The synthetic systems should support:

### Variable Dimensions

- N_S = number of synaptic input channels
- N_H = hidden dynamical dimension
- N_C = number of output channels

### Configurable Properties

- Sparsity
- Spectral radius
- Modularity
- Timescale diversity
- Coupling strength
- Output dimensionality

### Stability

Ensure $\rho(W_h) < 1$ (spectral radius) to prevent unstable exploding dynamics.

## 6. Input Generation System

### 6.1 Master Poisson Process

Generate a global Poisson event stream: $N(t) \sim \text{Poisson}(\lambda_{max})$

Each event is routed to an input channel.

### 6.2 Routing Distribution

Define $P \in \mathbb{R}^{N_S}$ with $\sum_i P_i = 1$. Each event selects channel i with probability P_i. This supports arbitrary N_S.

### 6.3 Synaptic Kernel

Convert event timestamps into smooth input currents. For each event time t_k:

$$\alpha(t) = H(t) \cdot \frac{t}{\tau_s} \cdot \exp\left(-\frac{t}{\tau_s}\right)$$

Where H(t) is the Heaviside step function and τ_s is the synaptic decay constant.

Each input channel becomes: $u_i(t) = \sum_k \alpha(t - t_k)$

### 6.4 Correlated Inputs (Future Extension)

Later versions may include latent shared processes, assembly activation, Cox processes, Hawkes processes, and oscillatory modulation.

Version 1 uses master Poisson + routing.

## 7. Simulation Engine

### Time Discretization

Use Euler integration initially.

**Discrete update:**

$$h[t+1] = h[t] + \frac{dt \cdot \left(-h[t] + \tanh(W_h h[t] + W_u u[t] + b_h)\right)}{\tau}$$

**Output:**

$$v[t] = W_o h[t] + b_o$$

## 8. Output Structure

Output tensor: $V \in \mathbb{R}^{T \times N_C}$

Where T = number of timesteps and N_C = output channels. Each channel behaves like a pseudo-compartment voltage trace.

## 9. Summary Statistics Layer W(V)

**Goal:** Map variable/high-dimensional traces into fixed-dimensional descriptors.

### 9.1 Initial Simple Statistics

**Per-channel:**

- Mean, variance, max, min, RMS
- Skewness, kurtosis

**Global statistics:**

- Pairwise correlations
- Covariance eigenvalues
- Spectral power
- Autocorrelation decay
- Entropy estimates

### 9.2 Spectral Statistics

FFT-based descriptors:

- Dominant frequencies
- Spectral centroid
- Spectral entropy
- Low/high frequency ratios

### 9.3 Dynamical Statistics

Potential future metrics: Lyapunov estimates, intrinsic dimensionality, memory capacity, controllability, observability, response separability.

## 10. System Descriptor S(F_θ)

The descriptor vector should summarize computational behavior, temporal structure, responsiveness, and complexity.

**Example:** $S(F_\theta) \in \mathbb{R}^K$ where $K$ is fixed.

## 11. Distance Metrics

**Goal:** $D(F_{\theta_1}, F_{\theta_2})$ measures functional dissimilarity.

### 11.1 Initial Metrics

**Euclidean:** $D = \|S_1 - S_2\|_2$

**Cosine:** $D = 1 - \text{cosine\_similarity}(S_1, S_2)$

**Mahalanobis:** $D = \sqrt{(S_1 - S_2)^T \Sigma^{-1} (S_1 - S_2)}$

### 11.2 Future Metrics

Possible advanced methods: Wasserstein distance, kernel MMD, operator norms, trajectory manifold distances, Koopman spectral distances, latent embedding distances.

## 12. Experimental Pipeline

### Pipeline Steps

1. Generate random system parameters θ
2. Generate stochastic input realization u(t)
3. Run simulation
4. Record outputs v(t)
5. Compute summary statistics W(V)
6. Generate descriptor vector S(F_θ)
7. Compare systems using D()

## 13. Initial Development Goals

### Goal 1

Verify stable simulations, realistic smooth traces, controllable complexity.

### Goal 2

Verify descriptors distinguish systems with different recurrent structure, different timescales, different connectivity.

### Goal 3

Test descriptor robustness across random input realizations.

### Goal 4

Visualize descriptor space using PCA, t-SNE, UMAP to determine whether similar systems cluster together.

## 14. Recommended Project Structure

```
project_root/
├── configs/
│   └── default.yaml
├── src/
│   ├── systems/
│   │   ├── recurrent_system.py
│   │   └── system_generator.py
│   ├── inputs/
│   │   ├── poisson_generator.py
│   │   └── synaptic_kernel.py
│   ├── simulation/
│   │   └── simulator.py
│   ├── summaries/
│   │   ├── statistics.py
│   │   ├── spectral.py
│   │   └── descriptors.py
│   ├── metrics/
│   │   └── distances.py
│   ├── experiments/
│   │   ├── compare_systems.py
│   │   └── sweep_parameters.py
│   └── visualization/
│       ├── plotting.py
│       └── embeddings.py
├── notebooks/
│   └── exploration.ipynb
└── tests/
    ├── test_inputs.py
    ├── test_systems.py
    └── test_statistics.py
```

## 15. Suggested Python Dependencies

**Core:** numpy, scipy, matplotlib

### Optional

scikit-learn, networkx, seaborn, umap-learn, pytorch (future)

## Initial Parameter Suggestions

### System Sizes

- N_S = 25 (input channels)
- N_H = 64 (hidden dimensions)
- N_C = 32 (output channels)

### Simulation

- dt = 1 ms
- T = 10 seconds

### Dynamics

- spectral_radius(W_h) = 0.8
- sparsity = 0.1
- $\tau \sim \text{Uniform}(10\text{ ms}, 100\text{ ms})$

### Input

- $\lambda_{max} = 100$ Hz
- $\tau_s = 10$ ms

## 17. Validation Strategy

The framework should recover known differences between sparse vs dense systems, fast vs slow systems, modular vs random systems, low vs high spectral radius systems before transitioning to morphology-based systems.

## 18. Long-Term Extensions

### Morphology

Replace synthetic recurrent systems with graph-based cable systems, Arbor simulations, SWC morphologies.

### Spiking Dynamics

Add thresholding, reset, spike-triggered adaptation.

### Plasticity

Allow topology changes, synapse growth/removal, parameter evolution.

### Learned Embeddings

Use autoencoders, contrastive learning, neural operators to learn functional embeddings automatically.

## 19. Core Scientific Question

The ultimate goal is: Can neuronal structures be compared in terms of their dynamical computational behavior, rather than only their geometry?

Or equivalently: Can we define a meaningful geometry on neural systems induced by their input-output dynamics?

## 20. Recommended First Milestone

Implement:

1. Random recurrent dynamical system
2. Master Poisson input generator
3. Synaptic kernel filtering
4. Simulation loop
5. Basic summary statistics
6. Euclidean distance comparisons
7. PCA visualization

Before adding morphology, spikes, plasticity, and nonlinear biological mechanisms.
