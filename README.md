# Neurosignature

A framework for **functional system identification of dynamical neural operators**.

The central product of this framework is the **system identification (SI) pipeline**: drive an arbitrary dynamical operator with stochastic synaptic-like inputs, compress its output traces into a fixed-dimensional descriptor, and compare operators via functional distance metrics. The dynamical system model is deliberately interchangeable — the current implementation uses a **Continuous-Time RNN (CTRNN)** as a lightweight, controllable test model. Future targets include graph-diffusion systems, cable-equation surrogates, and full morphological simulations (e.g., Arbor).

## Overview

The core scientific question: *Can neuronal structures be compared in terms of their dynamical computational behavior, rather than only their geometry?*

Or equivalently: *Can we define a meaningful geometry on neural systems induced by their input-output dynamics?*

### The Pipeline

Given a dynamical system $F_\theta$ parameterised by $\theta$:

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

### 1. Dynamical System Model (Current Test Model: CTRNN)

**Passive subthreshold dynamical operator** — a CTRNN used as a controllable surrogate for neuronal membrane dynamics during SI pipeline development. It is not the scientific focus; its role is to produce realistic-looking, parameterically varied traces for validating the pipeline.

**Hidden state dynamics:**

$$\frac{dh}{dt} = -\Lambda h + g \, W_{\text{prop}} \, \phi\!\left(W_{\text{int}} \, h\right) + W_u \, u$$

**Output equation:**

$$v(t) = V_{\text{rest}} + \eta(W_o \, h)$$

- **State:** $h(t) \in \mathbb{R}^{N_H}$ — deviations from equilibrium
- **Inputs:** $u(t) \in \mathbb{R}^{N_S}$ — synaptic drive channels
- **Outputs:** $v(t) \in \mathbb{R}^{N_C}$ — voltage-like traces around $V_{\text{rest}}$

Key properties:

- $-\Lambda h$ leak term drives activity back to baseline ($v \to V_{\text{rest}}$ when $u=0$)
- Split recurrent structure: $W_{\text{int}}$ (latent mixing) + $W_{\text{prop}}$ (propagation)
- $W_u u$ is **not** scaled by $g$, so passive systems ($g=0$) still respond to input
- Global gain $g$ controls recurrent regime: passive ($g \approx 0$–$0.1$), active ($g \approx 0.3$–$1.0$)
- Independent spectral radii $r_{\text{int}} = \rho(W_{\text{int}})$ and $r_{\text{prop}} = \rho(W_{\text{prop}})$
- Heterogeneous per-unit timescales $\tau_i \sim \text{LogUniform}(10\text{ ms},\, 100\text{ ms})$
- Polarity modes determined by $\eta$: **bipolar** ($\eta=\mathbf{1}$), **excitatory** ($\eta=+\text{ReLU}$), **inhibitory** ($\eta=-\text{ReLU}$)

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
- **Cosine:** $D = 1 - S_1 \cdot S_2 / \|S_1\|_2 \|S_2\|_2$
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
from neurosignature.systems import SystemGenerator
from neurosignature.inputs import PoissonGenerator, SynapticKernel
from neurosignature.simulation import Simulator

# Create a passive test system via the factory
generator = SystemGenerator(n_hidden=64, n_inputs=25, n_outputs=32)
system = generator.generate_passive_system(
    g=0.1,
    r_int=0.5,
    r_prop=0.1,
    sparsity=0.1,
    polarity="bipolar",
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

# Generate a diverse ensemble (random r_int, r_prop, g, sparsity)
generator = SystemGenerator(n_hidden=64, n_inputs=25, n_outputs=32)
systems = generator.generate_ensemble(n_systems=20)

# Compare pairwise using shared input realization
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
| $g$ | 0.1 | Global recurrent gain |
| $r_{\text{int}}$ | 0.5 | Spectral radius of $W_{\text{int}}$ |
| $r_{\text{prop}}$ | 0.1 | Spectral radius of $W_{\text{prop}}$ |
| sparsity | 0.1 | Recurrent matrix sparsity |
| $V_{\text{rest}}$ | −65 mV | Resting membrane potential |
| $\tau_i$ | LogUniform(10–100 ms) | Per-unit timescales |
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

The SI pipeline is validated progressively across three dynamical regimes of the test model:

| Regime | Gain $g$ | Recurrence | Expected behavior |
|--------|----------|------------|-------------------|
| **Passive** | 0–0.1 | weak | stable dissipative filtering, EPSP-like traces |
| **Weakly active** | 0.2–0.4 | moderate | nonlinear amplification, longer memory |
| **Strongly active** | 0.5–1.0 | dominant | oscillatory transients, high dynamical dimensionality |

Within each regime the framework is expected to recover known differences between:

- Sparse vs dense connectivity systems
- Fast vs slow timescale distributions
- Low vs high integration spectral radius ($r_{\text{int}}$)
- Low vs high recurrent gain ($g$)

See `notebooks/exploration.ipynb` for validation experiments.

## Long-Term Extensions

The CTRNN is a development scaffold. Once the SI pipeline is validated, the system model is intended to be replaced by progressively more realistic operators:

- **Graph-diffusion / cable surrogates:** Lightweight morphology-dependent operators
- **Arbor simulations:** Full multi-compartment biophysical models driven through the same pipeline
- **SWC morphologies:** Real reconstructed neuron geometries as the dynamical operator
- **Spiking dynamics:** Thresholding, reset, spike-triggered adaptation
- **Plasticity:** Topology changes, synapse growth/removal during simulation
- **Learned embeddings:** Autoencoders, contrastive learning, neural operators for functional geometry

## License

<!-- TODO: Add license -->
