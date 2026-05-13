# Passive vs Active Dynamical Modes

# CTRNN Operating Regimes

---

# 1. Purpose

The CTRNN framework supports multiple dynamical operating regimes.

These regimes are NOT different architectures.

They are different parameterizations of the same general dynamical system:

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

Different parameter choices produce qualitatively different behaviors:

- passive filtering,
- active amplification,
- or autonomous recurrent dynamics.

---

# 2. Core Distinction

The key balance is between:

- dissipation (leak)
vs
- amplification/recurrent feedback.

---

# 3. Passive Mode

## Goal

Model:

- passive membrane dynamics
- alpha synapse integration
- subthreshold cable filtering
- fading memory

The system should behave like:

- a stable dissipative filter.

---

# 3.1 Qualitative Properties

Desired behavior:

- perturbations decay naturally
- activity returns to baseline
- no spontaneous activity
- no oscillations
- no attractors
- smooth low-frequency traces
- approximately linear behavior

---

# 3.2 Biological Interpretation

Passive mode approximates:

- leak conductance
- membrane capacitance
- passive dendritic filtering
- alpha synapse integration
- electrotonic attenuation

---

# 3.3 Parameter Regime

## Leak Dominated

\[
\Lambda \gg gW_{\mathrm{prop}}
\]

Leak should dominate long-term behavior.

---

## Weak Recurrence

Recommended:

\[
g \in [0,0.2]
\]

Start with:

\[
g = 0
\]

or extremely small recurrence.

---

## Small Spectral Radius

Recommended:

\[
\rho(W_{\mathrm{prop}})
< 0.3
\]

Typical:

\[
0.05 - 0.2
\]

---

## Mild Nonlinearity

Hidden states should remain near:

- linear region of tanh.

This creates:

- weak nonlinear filtering
rather than:
- strong nonlinear amplification.

---

## Stable Equilibrium

Without input:

\[
u(t)=0
\]

the system should satisfy:

\[
h(t)\to 0
\]

and:

\[
v(t)\to V_{\mathrm{rest}}
\]

---

# 3.4 Expected Dynamics

Passive mode should produce:

- EPSP/IPSP-like fluctuations
- transient depolarizations
- smooth exponential decay
- short fading memory
- stable bounded traces

---

# 4. Active Mode

## Goal

Model:

- nonlinear dendritic processing
- active membrane mechanisms
- regenerative amplification
- richer dynamical computation

---

# 4.1 Qualitative Properties

Expected behavior:

- stronger nonlinear interactions
- longer temporal memory
- state-dependent amplification
- richer latent dynamics
- possible oscillatory modes
- increased sensitivity to input timing

---

# 4.2 Biological Interpretation

Active mode approximates:

- active dendrites
- voltage-gated conductances
- NMDA amplification
- nonlinear branch interactions
- regenerative membrane dynamics

---

# 4.3 Parameter Regime

## Reduced Leak Dominance

Leak still stabilizes the system,
but recurrence becomes significant.

---

## Stronger Recurrence

Recommended:

\[
g \in [0.3,1.0]
\]

---

## Larger Spectral Radius

Recommended:

\[
\rho(W_{\mathrm{prop}})
\in [0.5,1.0]
\]

---

## Stronger Nonlinear Operation

Hidden states frequently enter:

- nonlinear/saturating activation regions.

This produces:

- state-dependent responses
- nonlinear mode coupling

---

## Long-Lived Internal Modes

Past input persists longer.

The system develops:

- richer memory structure
- stronger latent interactions

---

# 4.4 Expected Dynamics

Active mode may produce:

- nonlinear amplification
- prolonged depolarizations
- oscillatory transients
- complex autocorrelation structure
- increased dynamical dimensionality

---

# 5. Autonomous / Unstable Regime

## Warning

If recurrent amplification overwhelms leak:

\[
g\rho(W_{\mathrm{prop}})
\gg
\Lambda
\]

the system may enter:

- unstable
- self-sustaining
- chaotic
- oscillatory regimes.

---

# 5.1 Characteristics

Possible behaviors:

- spontaneous activity
- attractors
- oscillations
- runaway amplification
- persistent internal dynamics

---

# 5.2 Interpretation

This regime resembles:

- reservoir computing systems
- recurrent AI models
- cortical network models

NOT:

- passive membrane dynamics.

Avoid this regime during early development.

---

# 6. Summary Table

| Regime | Leak | Recurrence | Memory | Behavior |
|---|---|---|---|---|
| Passive | dominant | weak | short fading | stable filtering |
| Active | balanced | moderate | longer | nonlinear amplification |
| Autonomous | weak | dominant | persistent | self-sustained dynamics |

---

# 7. Recommended Development Path

---

## Phase 1 — Passive

Start with:

- passive mode only.

Goal:

- validate SI pipeline
- validate summary statistics
- validate distance metrics

Recommended:

\[
g \approx 0-0.1
\]

---

## Phase 2 — Weakly Active

Introduce:

- moderate recurrence
- weak nonlinear amplification

Goal:

- richer operator diversity
- stronger functional differences

Recommended:

\[
g \approx 0.2-0.4
\]

---

## Phase 3 — Strongly Active

Explore:

- active nonlinear dynamics
- long-memory systems
- oscillatory transients

Goal:

- test robustness of SI descriptors

---

# 8. Conceptual Interpretation

The CTRNN framework should be interpreted as:

a continuum of dynamical operator families

ranging from:

- passive dissipative filters
to:
- active nonlinear recurrent systems.

Different operating regimes correspond to:

- different balances of dissipation and amplification,
not:
- fundamentally different architectures.

```
