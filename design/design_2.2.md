# CTRNN Design Update

# Passive Subthreshold Dynamical Operator Framework

---

# 1. Purpose

This CTRNN framework is intended to serve as a lightweight surrogate for:

- passive neuronal membrane dynamics,
- dendritic filtering,
- and morphology-dependent dynamical operators.

The immediate goal is NOT biological realism.

The goal is to develop and validate:

- system identification (SI) methods,
- functional distance metrics,
- and low-dimensional dynamical descriptors.

The system should produce:

- smooth voltage-like traces,
- transient perturbations around rest,
- fading memory,
- stable bounded dynamics.

---

# 2. Core Philosophy

The model should behave like:

- a passive dissipative filter system,

NOT:

- an autonomous recurrent neural network,
- a reservoir computer,
- or a chaotic dynamical system.

Desired behavior:

- synaptic input perturbs the system
- perturbations decay naturally
- activity returns to resting baseline
- no self-sustained activity
- no spikes
- no attractors
- no oscillatory instability

---

# 3. System Variables

## Inputs

u(t) ∈ R^(N_S)

Represents:

- synaptic drive channels

N_S:

- number of synaptic inputs

---

## Hidden State

h(t) ∈ R^(N_H)

Represents:

- latent dynamical modes
- internal filtering states
- pseudo-compartment states

NOT:

- biological neurons

N_H:

- hidden dynamical dimension

---

## Outputs

v(t) ∈ R^(N_C)

Represents:

- voltage-like traces
- pseudo-compartment observations

N_C:

- number of output channels

---

# 4. General Dynamical Model

The updated CTRNN dynamics are:

\[
\frac{dh}{dt}
=

-\Lambda h
+
g\,W_{\mathrm{prop}}
\phi\!\left(
W_{\mathrm{int}} h
+
W_u u(t)
\right)
\]

Outputs:

\[
v(t)
=

V_{\mathrm{rest}}
+
W_o h(t)
\]

---

# 5. Interpretation of Terms

---

## Leak Term

\[
-\Lambda h
\]

Represents:

- membrane leak
- passive relaxation
- dissipation

Ensures:

- activity decays toward baseline
- stability
- fading memory

---

## Integration Matrix

\[
W_{\mathrm{int}}
\]

Controls:

- latent mixing before nonlinearity
- local integration structure

Interpretation:

- pseudo-dendritic integration
- local coupling before saturation

---

## Nonlinearity

\[
\phi(\cdot)
\]

Recommended:

- tanh
- softplus

Purpose:

- bounded responses
- smooth saturation
- nonlinear filtering

---

## Propagation Matrix

\[
W_{\mathrm{prop}}
\]

Controls:

- propagation of transformed activity
- latent coupling between modes

Interpretation:

- passive propagation
- pseudo-cable coupling
- dynamical diffusion structure

---

## Input Projection Matrix

\[
W_u
\]

Maps:

- synaptic input channels
into:
- latent dynamical modes

---

## Output Projection Matrix

\[
W_o
\]

Maps:

- hidden state
into:
- observable voltage traces

---

## Global Gain

\[
g
\]

Controls:

- overall recurrent strength

Should remain small.

---

# 6. Resting Potential Dynamics

The hidden state represents:

- deviations from equilibrium.

If:

u(t) = 0

then:

h(t) -> 0

and therefore:

v(t) -> V_rest

This is a critical design requirement.

---

# 7. Memory Interpretation

The model should exhibit:

- fading memory only.

Memory arises from:

- leak timescales
- synaptic filtering
- weak recurrent coupling

NOT from:

- reverberatory attractors
- autonomous recurrence
- persistent internal activity

The system should resemble:

- passive membrane integration,
not:
- a reservoir computer.

---

# 8. Synaptic Input Generation

---

# 8.1 Master Poisson Process

Generate:

- a global Poisson event stream

\[
N(t) \sim \mathrm{Poisson}(\lambda_{\max})
\]

Each event is routed into:

- one synaptic input channel.

---

# 8.2 Routing Distribution

Define:

\[
P \in \mathbb{R}^{N_S}
\]

with:

\[
\sum_i P_i = 1
\]

Each event selects input channel i with probability P_i.

This supports:

- arbitrary N_S
- normalized total input drive

---

# 8.3 Synaptic Kernel

Convert event timestamps into smooth currents.

Recommended alpha kernel:

\[
\alpha(t)
=

H(t)\frac{t}{\tau_s}e^{-t/\tau_s}
\]

Each input channel becomes:

\[
u_i(t)
=

\sum_k \alpha(t-t_k^{(i)})
\]

This provides:

- smooth EPSP-like dynamics
- temporal memory
- realistic filtering

---

# 9. Polarity Modes

---

# 9.1 Excitatory-Only

Outputs remain above resting potential.

\[
v(t)
=

V_{\mathrm{rest}}
+
\mathrm{softplus}(W_o h(t))
\]

Recommended:

- positive W_u
- positive W_o

Produces:

- depolarizing-only fluctuations

---

# 9.2 Inhibitory-Only

Outputs remain below resting potential.

\[
v(t)
=

V_{\mathrm{rest}}
-

\mathrm{softplus}(W_o h(t))
\]

Produces:

- hyperpolarizing-only fluctuations

---

# 9.3 Bipolar

Both positive and negative perturbations allowed.

\[
v(t)
=

V_{\mathrm{rest}}
+
W_o h(t)
\]

Most general mode.

---

# 10. Matrix Sampling Strategy

The matrices define:

- the geometry of the dynamical operator space.

Sampling should therefore be:

- stable
- structured
- controllable

NOT arbitrary.

---

# 10.1 Integration Matrix

\[
W_{\mathrm{int}}
\]

Recommended:

- sparse Gaussian random matrix

Sample:

\[
(W_{\mathrm{int}})_{ij}
\sim
\mathcal{N}(0,\sigma^2)
\]

Apply:

- sparsity mask

Normalize spectral radius:

\[
W \leftarrow \frac{r}{\rho(W)}W
\]

Suggested:

\[
r_{\mathrm{int}} \in [0.3,0.8]
\]

---

# 10.2 Propagation Matrix

\[
W_{\mathrm{prop}}
\]

Recommended:

- sparse symmetric matrix

Example:

\[
W_{\mathrm{prop}}
=

\frac{A+A^T}{2}
\]

where:

- A is sparse Gaussian.

Suggested:

\[
r_{\mathrm{prop}} \in [0.05,0.3]
\]

This keeps recurrence:

- weak
- smooth
- diffusion-like

---

# 10.3 Input Matrix

\[
W_u
\]

Excitatory-only:

- positive weights only

Example:

\[
(W_u)_{ij}
\sim
\mathrm{Uniform}(0,1)
\]

---

# 10.4 Output Matrix

\[
W_o
\]

Maps latent states into:

- voltage observations

Typically:

- dense
- small magnitude

---

# 11. Spectral Radius Control

For any recurrent matrix:

Compute:

\[
\rho(W)
=

\max |\lambda_i|
\]

Rescale:

\[
W
\leftarrow
\frac{r}{\rho(W)}W
\]

This controls:

- memory depth
- stability
- temporal richness

---

# 12. Recommended Initial Parameters

---

## Dimensions

N_S = 25

N_H = 64

N_C = 32

---

## Simulation

dt = 1 ms

T = 10 seconds

---

## Dynamics

g = 0.1–0.3

r_int = 0.5

r_prop = 0.1

---

## Timescales

tau_i ~ LogUniform(10 ms, 100 ms)

---

## Synaptic Filtering

tau_s = 10 ms

---

## Voltage

V_rest = -65 mV

---

# 13. Recommended Initial Regime

Start with:

g = 0

or extremely small recurrence.

This allows validation of:

- synaptic filtering
- descriptor pipeline
- SI distances

before introducing:

- latent recurrent structure.

---

# 14. System Identification Validation

The synthetic systems provide:

- controllable operator families
for testing SI metrics.

Validation should compare:

- SI distances
vs
- matrix/operator distances

BUT:

the ultimate goal is NOT:

- recovering raw matrix differences

Instead the goal is:

- recovering functional/operator similarity.

---

# 15. Recommended Validation Levels

---

## Level 1

Compare SI distance to:

\[
\|W_1-W_2\|_F
\]

(sanity check only)

---

## Level 2

Compare SI distance to:

- spectral distances
- eigenvalue structure
- timescale similarity

---

## Level 3

Test:

- different matrices
- similar dynamics

A successful SI metric should identify:

- functional equivalence,
not merely:
- parameter similarity.

---

# 16. Long-Term Direction

Eventually replace synthetic CTRNN systems with:

- graph diffusion systems
- cable-equation surrogates
- Arbor simulations
- morphology-dependent operators

The current framework serves as:

- a controlled dynamical testbed
for developing:
- system identification
- operator embeddings
- functional geometry metrics.

```
