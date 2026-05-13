"""Simulation engine for running dynamical systems."""

import numpy as np
import pynapple as nap
from typing import Any, List, Optional, Tuple, Union


class Simulator:
    """Simulation engine for running dynamical systems forward in time.

    Uses Euler integration with system-provided step dynamics:
    h[t+1] = system.step(h[t], u[t], dt)

    The system must implement:
    - step(h, u, dt): Compute next hidden state
    - compute_output(h): Compute output from hidden state
    - reset_state(): Return initial hidden state
    - n_hidden, n_inputs, n_outputs: Int properties

    Args:
        system: Dynamical system with step() interface
        dt_ms: Integration time step in milliseconds (default: 1.0)
    """

    def __init__(
        self,
        system: Any,
        dt_ms: float = 1.0,
    ):
        self.system = system
        self.dt_ms = dt_ms

    def run(
        self,
        inputs: Union[np.ndarray, nap.TsdFrame],
        initial_state: Optional[np.ndarray] = None,
        record_states: bool = False,
    ) -> Tuple[nap.TsdFrame, Optional[nap.TsdFrame]]:
        """Run simulation with given input sequence.

        Args:
            inputs: Input current matrix, shape (n_timesteps, n_inputs).
                May be a plain numpy array or a pynapple ``TsdFrame``.
            initial_state: Initial hidden state. If None, uses zeros.
            record_states: Whether to record all hidden states

        Returns:
            outputs: ``TsdFrame`` of output traces (n_timesteps, n_outputs)
                in mV, time-indexed in ms.
            states: ``TsdFrame`` of hidden states if record_states=True,
                else None.
        """
        inputs_arr = np.asarray(inputs)
        n_steps = inputs_arr.shape[0]

        # Initialize state
        if initial_state is None:
            h = self.system.reset_state()
        else:
            h = initial_state.copy()

        # Storage
        outputs_arr = np.zeros((n_steps, self.system.n_outputs))
        n_hid = self.system.n_hidden
        states_arr = np.zeros((n_steps, n_hid)) if record_states else None

        # Integration loop
        for t in range(n_steps):
            u = inputs_arr[t]

            # Record state before step (optional)
            if record_states:
                states_arr[t] = h.copy()

            # Compute output
            outputs_arr[t] = self.system.compute_output(h)

            # Update state
            h = self.system.step(h, u, self.dt_ms)

        t_ms = np.arange(n_steps, dtype=float) * self.dt_ms
        outputs = nap.TsdFrame(t=t_ms, d=outputs_arr, time_units="ms")
        states: Optional[nap.TsdFrame] = (
            nap.TsdFrame(t=t_ms, d=states_arr, time_units="ms")
            if states_arr is not None
            else None
        )
        return outputs, states

    def run_batch(
        self,
        input_batch: np.ndarray,
        initial_states: Optional[np.ndarray] = None,
    ) -> List[nap.TsdFrame]:
        """Run simulation for batch of input sequences (same system).

        Args:
            input_batch: Batch of inputs,
                shape (n_samples, n_timesteps, n_inputs)
            initial_states: Initial states,
                shape (n_samples, n_hidden) or None

        Returns:
            List of ``TsdFrame`` outputs, one per sample.
            Each TsdFrame has shape (n_timesteps, n_outputs).
        """
        n_samples = input_batch.shape[0]
        outputs = []

        for i in range(n_samples):
            initial = None
            if initial_states is not None:
                initial = initial_states[i]

            out, _ = self.run(input_batch[i], initial_state=initial)
            outputs.append(out)

        return outputs

    def run_with_multiple_systems(
        self,
        systems: list,
        inputs: Union[np.ndarray, nap.TsdFrame],
    ) -> List[nap.TsdFrame]:
        """Run same input through multiple different systems.

        Args:
            systems: List of systems with step() interface
            inputs: Input currents, shape (n_timesteps, n_inputs)

        Returns:
            List of ``TsdFrame`` output traces, one per system.
            Each TsdFrame has shape (n_timesteps, n_outputs).
        """
        all_outputs = []
        for system in systems:
            outputs, _ = self._run_single(system, inputs)
            all_outputs.append(outputs)

        return all_outputs

    def _run_single(
        self,
        system: Any,
        inputs: Union[np.ndarray, nap.TsdFrame],
        initial_state: Optional[np.ndarray] = None,
    ) -> Tuple[nap.TsdFrame, None]:
        """Run simulation for a single system (internal helper)."""
        inputs_arr = np.asarray(inputs)
        n_steps = inputs_arr.shape[0]

        if initial_state is None:
            h = system.reset_state()
        else:
            h = initial_state.copy()

        outputs_arr = np.zeros((n_steps, system.n_outputs))

        for t in range(n_steps):
            outputs_arr[t] = system.compute_output(h)
            h = system.step(h, inputs_arr[t], self.dt_ms)

        t_ms = np.arange(n_steps, dtype=float) * self.dt_ms
        return nap.TsdFrame(t=t_ms, d=outputs_arr, time_units="ms"), None

    def check_stability(
        self,
        inputs: np.ndarray,
        max_value: float = 100.0,
    ) -> bool:
        """Check if simulation remains stable (bounded).

        Args:
            inputs: Input currents
            max_value: Maximum allowed absolute value for outputs

        Returns:
            True if stable, False if outputs exceed bounds
        """
        outputs, _ = self.run(inputs)
        return np.all(np.abs(outputs) < max_value)

    def compute_stationary_stats(
        self,
        inputs: np.ndarray,
        burn_in_steps: int = 100,
    ) -> dict:
        """Compute statistics of outputs after burn-in period.

        Args:
            inputs: Input currents
            burn_in_steps: Number of initial steps to discard

        Returns:
            Dictionary with mean, std, min, max per output channel
        """
        outputs, _ = self.run(inputs)

        # Discard burn-in
        outputs_stable = outputs[burn_in_steps:]

        return {
            "mean": np.mean(outputs_stable, axis=0),
            "std": np.std(outputs_stable, axis=0),
            "min": np.min(outputs_stable, axis=0),
            "max": np.max(outputs_stable, axis=0),
        }
