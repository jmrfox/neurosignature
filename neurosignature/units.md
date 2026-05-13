# Unit System

This document defines the canonical unit conventions for the `neurosignature` package.
All public APIs must follow these conventions unless explicitly noted otherwise.

## Canonical Units

| Quantity | Unit | Symbol | Notes |
|---|---|---|---|
| Time | millisecond | ms | All `dt_ms`, `duration_ms`, `tau_ms` parameters |
| Membrane potential / voltage | millivolt | mV | `V_rest`, simulator outputs |
| Input current | nanoampere | nA | Input arrays fed to `Simulator.run()` |
| Firing rate | hertz | Hz | `rate_hz`, `lambda_max` in Poisson generators |
| Frequency (spectral) | hertz | Hz | FFT frequency axes, band definitions |
| Timescale / time constant | millisecond | ms | `tau_ms` in `SynapticKernel`, `ContinuousTimeRNN` |

## Pynapple Time Index Convention

`pynapple` time series objects (`TsdFrame`, `Ts`, `TsGroup`) are created with
`time_units="ms"` so that the constructor receives values already in milliseconds.
Internally pynapple normalises timestamps to **seconds** (its native representation),
so `.index` on any pynapple object will show seconds.

When constructing pynapple objects in this codebase always pass `time_units="ms"`:

```python
import pynapple as nap
import numpy as np

t_ms = np.arange(n_steps, dtype=float) * dt_ms  # ms
tsd = nap.TsdFrame(t=t_ms, d=data, time_units="ms")
# tsd.index is in seconds internally
```

## Parameter Naming Conventions

- Parameters carrying a time value are suffixed `_ms` (e.g. `dt_ms`, `duration_ms`, `tau_ms`).
- Parameters carrying a rate are suffixed `_hz` (e.g. `rate_hz`).
- Parameters carrying a voltage are suffixed `_mv` only when ambiguity exists; `V_rest` is self-explanatory.
- Avoid mixing `_s` (seconds) suffixes — the only time unit in this codebase is milliseconds.
