"""Passive subthreshold dynamical operator framework."""

import numpy as np
from typing import Union, Callable
from neurosignature.math import relu, softplus


class ContinuousTimeRNN:
    """Passive subthreshold dynamical operator with split recurrent structure.

    Dynamics: dh/dt = -Λh + g * W_prop @ φ(W_int @ h) + W_u @ u
    Output: v = V_rest + polarity_transform(W_o @ h)

    This implements a dissipative filtering system rather than an autonomous
    recurrent network, with resting potential-centered dynamics.
    Note: the input term W_u @ u is not scaled by g, enabling passive
    (g=0) systems to still respond to inputs.

    All weight matrices must be provided explicitly. Use SystemGenerator
    for convenient matrix initialization methods.

    Args:
        n_hidden: Number of hidden state dimensions (N_H)
        n_inputs: Number of input channels (N_S)
        n_outputs: Number of output channels (N_C)
        W_int: Integration matrix (N_H x N_H)
        W_prop: Propagation matrix (N_H x N_H)
        W_u: Input projection matrix (N_H x N_S)
        W_o: Output projection matrix (N_C x N_H)
        g: Global recurrent gain (default: 0.1)
        tau: Timescale(s) for hidden state decay. Scalar or array of
            shape (n_hidden,). Default: 50.0 ms
        V_rest: Resting membrane potential in mV (default: -65.0)
        polarity: Output polarity mode - "bipolar", "excitatory",
            or "inhibitory"
        phi: Nonlinearity - "tanh" or "softplus"
    """

    def __init__(
        self,
        n_hidden: int,
        n_inputs: int,
        n_outputs: int,
        W_int: np.ndarray,
        W_prop: np.ndarray,
        W_u: np.ndarray,
        W_o: np.ndarray,
        g: float = 0.1,
        tau: Union[float, np.ndarray] = 50.0,
        V_rest: float = -65.0,
        polarity: str = "bipolar",
        phi: str = "tanh",
    ):
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.g = g
        self.V_rest = V_rest
        self.polarity = polarity.lower()
        self.phi_name = phi.lower()

        # Validate polarity
        if self.polarity not in ["bipolar", "excitatory", "inhibitory"]:
            raise ValueError(
                f"Invalid polarity: {polarity}. "
                "Use 'bipolar', 'excitatory', or 'inhibitory'"
            )

        # Validate phi
        if self.phi_name not in ["tanh", "softplus"]:
            raise ValueError(f"Invalid phi: {phi}. Use 'tanh' or 'softplus'")

        # Set nonlinearity function
        if self.phi_name == "tanh":
            self.phi: Callable = np.tanh
        else:
            self.phi = softplus  # Use imported softplus function

        # Set timescale
        if np.isscalar(tau):
            self.tau = np.full(n_hidden, tau)
        else:
            self.tau = np.asarray(tau)

        # Validate and store weight matrices
        self.W_int = np.asarray(W_int)
        if self.W_int.shape != (n_hidden, n_hidden):
            raise ValueError(
                f"W_int shape {self.W_int.shape} != ({n_hidden}, {n_hidden})"
            )

        self.W_prop = np.asarray(W_prop)
        if self.W_prop.shape != (n_hidden, n_hidden):
            raise ValueError(
                f"W_prop shape {self.W_prop.shape} != ({n_hidden}, {n_hidden})"
            )

        self.W_u = np.asarray(W_u)
        if self.W_u.shape != (n_hidden, n_inputs):
            raise ValueError(f"W_u shape {self.W_u.shape} != ({n_hidden}, {n_inputs})")

        self.W_o = np.asarray(W_o)
        if self.W_o.shape != (n_outputs, n_hidden):
            raise ValueError(f"W_o shape {self.W_o.shape} != ({n_outputs}, {n_hidden})")

    def step(self, h: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Single Euler integration step.

        Args:
            h: Current hidden state, shape (n_hidden,)
            u: Current input, shape (n_inputs,)
            dt: Time step size

        Returns:
            Updated hidden state, shape (n_hidden,)
        """
        # Leak term: -Λh where Λ = 1/tau
        leak = -h / self.tau

        # Input term: W_u @ u (direct drive, not scaled by g)
        input_term = self.W_u @ u

        # Recurrent term: g * W_prop @ φ(W_int @ h)
        recurrent = self.g * (self.W_prop @ self.phi(self.W_int @ h))

        # Combined dynamics: dh/dt = -Λh + W_u @ u + g * W_prop @ φ(W_int @ h)
        dh = leak + input_term + recurrent
        h_new = h + dt * dh

        return h_new

    def compute_output(
        self, h: np.ndarray, polarity_function: str = "relu"
    ) -> np.ndarray:
        """Compute output from hidden state with polarity transform.

        Args:
            h: Hidden state, shape (n_hidden,)

        Returns:
            Output voltage, shape (n_outputs,)
        """
        # Base output projection
        projection = self.W_o @ h

        pfxn_map = {"relu": relu, "softplus": softplus}

        # Apply polarity transform
        if self.polarity == "excitatory":
            perturbation = pfxn_map[polarity_function](projection)
        elif self.polarity == "inhibitory":
            perturbation = -pfxn_map[polarity_function](projection)
        else:
            # bipolar: linear
            perturbation = projection

        return self.V_rest + perturbation

    def reset_state(self) -> np.ndarray:
        """Return initial hidden state (zeros)."""
        return np.zeros(self.n_hidden)

    def get_parameters(self) -> dict:
        """Return dictionary of all system parameters."""
        return {
            "W_int": self.W_int,
            "W_prop": self.W_prop,
            "W_u": self.W_u,
            "W_o": self.W_o,
            "g": self.g,
            "tau": self.tau,
            "V_rest": self.V_rest,
            "polarity": self.polarity,
            "phi": self.phi_name,
            "n_hidden": self.n_hidden,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }
