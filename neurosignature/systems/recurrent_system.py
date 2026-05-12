"""Continuous-time recurrent neural dynamical system."""

import numpy as np
from typing import Optional, Union


class ContinuousTimeRNN:
    """Continuous-time recurrent neural network with tanh nonlinearity.

    Dynamics: dh/dt = (-h + tanh(W_h @ h + W_u @ u + b_h)) / tau
    Output: v = W_o @ h + b_o

    Args:
        n_hidden: Number of hidden state dimensions (N_H)
        n_inputs: Number of input channels (N_S)
        n_outputs: Number of output channels (N_C)
        tau: Timescale(s) for hidden state decay. Scalar or array of shape (n_hidden,)
        spectral_radius: Target spectral radius for recurrent weights (default: 0.8)
        sparsity: Sparsity level for recurrent weights [0, 1] (default: 0.1)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_hidden: int,
        n_inputs: int,
        n_outputs: int,
        tau: Optional[Union[float, np.ndarray]] = None,
        spectral_radius: float = 0.8,
        sparsity: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity

        # Initialize random number generator
        self.rng = np.random.default_rng(seed)

        # Set timescale (default: uniform 10-100ms range)
        if tau is None:
            self.tau = self.rng.uniform(10.0, 100.0, size=n_hidden)
        elif np.isscalar(tau):
            self.tau = np.full(n_hidden, tau)
        else:
            self.tau = np.asarray(tau)

        # Initialize weights and biases
        self.W_h = self._initialize_recurrent_weights()
        self.W_u = self.rng.normal(0, 1.0 / np.sqrt(n_inputs), (n_hidden, n_inputs))
        self.W_o = self.rng.normal(0, 1.0 / np.sqrt(n_hidden), (n_outputs, n_hidden))
        self.b_h = np.zeros(n_hidden)
        self.b_o = np.zeros(n_outputs)

    def _initialize_recurrent_weights(self) -> np.ndarray:
        """Initialize sparse recurrent weights with controlled spectral radius."""
        # Create sparse mask
        mask = self.rng.random((self.n_hidden, self.n_hidden)) < self.sparsity

        # Initialize weights from normal distribution
        W = self.rng.normal(0, 1.0, (self.n_hidden, self.n_hidden))
        W = W * mask

        # Scale to target spectral radius
        if np.any(mask):
            current_radius = np.max(np.abs(np.linalg.eigvals(W)))
            if current_radius > 0:
                W = W * (self.spectral_radius / current_radius)

        return W

    def step(self, h: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Single Euler integration step.

        Args:
            h: Current hidden state, shape (n_hidden,)
            u: Current input, shape (n_inputs,)
            dt: Time step size

        Returns:
            Updated hidden state, shape (n_hidden,)
        """
        # Compute pre-activation
        pre_activation = self.W_h @ h + self.W_u @ u + self.b_h

        # Euler step
        dh = (-h + np.tanh(pre_activation)) / self.tau
        h_new = h + dt * dh

        return h_new

    def compute_output(self, h: np.ndarray) -> np.ndarray:
        """Compute output from hidden state.

        Args:
            h: Hidden state, shape (n_hidden,)

        Returns:
            Output, shape (n_outputs,)
        """
        return self.W_o @ h + self.b_o

    def reset_state(self) -> np.ndarray:
        """Return initial hidden state (zeros)."""
        return np.zeros(self.n_hidden)

    def get_parameters(self) -> dict:
        """Return dictionary of all system parameters."""
        return {
            "W_h": self.W_h,
            "W_u": self.W_u,
            "W_o": self.W_o,
            "b_h": self.b_h,
            "b_o": self.b_o,
            "tau": self.tau,
            "n_hidden": self.n_hidden,
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
        }
