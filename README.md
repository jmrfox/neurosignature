# Neurosignature

A framework for **functional system identification of dynamical neural operators**.

This framework generates synthetic recurrent dynamical systems, drives them with stochastic synaptic-like inputs, and compresses their outputs into fixed-dimensional descriptors that enable functional comparison of systems via distance metrics.

## Overview

The core scientific question: *Can neuronal structures be compared in terms of their dynamical computational behavior, rather than only their geometry?*

Or equivalently: *Can we define a meaningful geometry on neural systems induced by their input-output dynamics?*

### The Pipeline

Given system parameters $\theta$, the framework implements:

```
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

## Core Features

### 1. Dynamical System Model

**Continuous-time recurrent neural network** with controlled stability:

$$\frac{dh}{dt} = \frac{-h + \tanh(W_h h + W_u u + b_h)}{\tau}$$

- **State:** $h(t) \in \mathbb{R}^{N_H}$
- **Inputs:** $u(t) \in \mathbb{R}^{N_S}$
- **Outputs:** $v(t) = W_o h + b_o \in \mathbb{R}^{N_C}$

Key properties:

- Configurable spectral radius $\rho(W_h) < 1$ (guarantees stability)
- Sparse recurrent connectivity
- Heterogeneous timescales $\tau \sim \text{Uniform}(10\text{ms}, 100\text{ms})$

### 2. Input Generation

**Master Poisson process** with synaptic kernel filtering:

- Global event stream: $N(t) \sim \text{Poisson}(\lambda_{max})$
- Channel routing: $P \in \mathbb{R}^{N_S}$ with $\sum_i P_i = 1$
- **Alpha synaptic kernel:**
  $$\alpha(t) = H(t) \cdot \frac{t}{\tau_s} \cdot \exp\left(-\frac{t}{\tau_s}\right)$$

### 3. Summary Statistics & Descriptors

Multi-level descriptor computation:

**Per-channel statistics:**

- Mean, variance, RMS, skewness, kurtosis

**Global statistics:**

- Pairwise correlations
- Covariance eigenvalues
- Autocorrelation decay
- Approximate entropy

**Spectral features:**

- Dominant frequencies (FFT)
- Spectral centroid and entropy
- Frequency band power ratios

All concatenated into fixed-length descriptor vector $S(F_\theta) \in \mathbb{R}^K$.

### 4. Distance Metrics

- **Euclidean:** $D = \|S_1 - S_2\|_2$
- **Cosine:** $D = 1 - \text{cosine\_similarity}(S_1, S_2)$
- **Mahalanobis:** $D = \sqrt{(S_1 - S_2)^T \Sigma^{-1} (S_1 - S_2)}$

### 5. Visualization

- PCA, t-SNE, UMAP embeddings of descriptor space
- Distance matrix heatmaps
- Trace visualization

## Architecture

```
neurosignature/
├── systems/              # Dynamical system models
│   ├── recurrent_system.py      # ContinuousTimeRNN
│   └── system_generator.py      # Factory for generating ensembles
├── inputs/               # Input generation
│   ├── poisson_generator.py     # Master Poisson + routing
│   └── synaptic_kernel.py       # Alpha function filtering
├── simulation/           # Simulation engine
│   └── simulator.py             # Euler integration
├── summaries/            # Descriptor computation
│   ├── statistics.py            # Basic & global statistics
│   ├── spectral.py              # FFT-based features
│   └── descriptors.py           # Descriptor assembly
├── metrics/              # Distance computation
│   └── distances.py
├── experiments/          # Experimental pipelines
│   ├── compare_systems.py       # Pairwise system comparison
│   └── sweep_parameters.py      # Parameter sensitivity
└── visualization/        # Plotting & embeddings
    ├── plotting.py
    └── embeddings.py
```

## Quick Start

### Generate a System and Simulate

```python
from neurosignature.systems import ContinuousTimeRNN
from neurosignature.inputs import PoissonGenerator, SynapticKernel
from neurosignature.simulation import Simulator

# Create system
system = ContinuousTimeRNN(
    n_hidden=64,
    n_inputs=25,
    n_outputs=32,
    spectral_radius=0.8,
    sparsity=0.1,
    seed=42,
)

# Generate Poisson input
gen = PoissonGenerator(n_channels=25, lambda_max=100.0, seed=42)
events = gen.generate_events(duration_ms=10000.0, dt_ms=1.0)

# Convert to synaptic currents
kernel = SynapticKernel(tau_s=10.0, dt_ms=1.0)
inputs = kernel.generate_input_currents(events, duration_ms=10000.0)

# Simulate
sim = Simulator(system, dt_ms=1.0)
outputs, states = sim.run(inputs)
print(f"Output shape: {outputs.shape}")  # (10000, 32)
```

### Compute Descriptors

```python
from neurosignature.summaries import DescriptorAssembler

assembler = DescriptorAssembler()
descriptor = assembler.compute_descriptor(outputs, dt_ms=1.0)
print(f"Descriptor dimension: {len(descriptor)}")
```

### Compare Systems

```python
from neurosignature.systems import SystemGenerator
from neurosignature.experiments import SystemComparator

# Generate ensemble
generator = SystemGenerator()
systems = generator.generate_ensemble(n_systems=20)

# Compare
comparator = SystemComparator(sim, assembler)
result = comparator.compare_with_shared_input(systems, inputs)
print(f"Distance matrix shape: {result['distance_matrix'].shape}")
```

### Visualize

```python
from neurosignature.visualization import compute_pca, plot_embedding

embedded, pca = compute_pca(result['descriptors'])
fig = plot_embedding(embedded, title="System Descriptor Space (PCA)")
```

See `notebooks/exploration.ipynb` for a complete walkthrough.

## Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N_S$ | 25 | Input channels |
| $N_H$ | 64 | Hidden dimensions |
| $N_C$ | 32 | Output channels |
| $\rho(W_h)$ | 0.8 | Spectral radius |
| sparsity | 0.1 | Recurrent sparsity |
| $dt$ | 1 ms | Integration step |
| $T$ | 10 s | Simulation duration |
| $\lambda_{max}$ | 100 Hz | Poisson rate |
| $\tau_s$ | 10 ms | Synaptic decay |

## Installation

<!-- TODO: Add installation instructions -->
```bash
# Placeholder - to be added
```

## Development

<!-- TODO: Add development setup -->
```bash
# Placeholder - to be added
```

## Testing

```bash
uv run pytest tests/ -v
```

## Validation

The framework recovers known differences between:

- Sparse vs dense connectivity systems
- Fast vs slow timescale systems
- Low vs high spectral radius systems

See `notebooks/exploration.ipynb` for validation experiments.

## Long-Term Extensions

- **Morphology:** Graph-based cable systems, Arbor simulations, SWC morphologies
- **Spiking:** Thresholding, reset, spike-triggered adaptation
- **Plasticity:** Topology changes, synapse growth/removal
- **Learned embeddings:** Autoencoders, contrastive learning

## License

<!-- TODO: Add license -->
