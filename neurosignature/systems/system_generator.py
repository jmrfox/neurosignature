"""Factory for generating dynamical systems with controlled variation."""

import numpy as np
from typing import Optional, List
from .recurrent_system import ContinuousTimeRNN


class SystemGenerator:
    """Generate collections of dynamical systems with controlled variation.

    Supports generating systems with different:
    - Spectral radius (r_int, r_prop)
    - Sparsity levels
    - Timescale distributions
    - Polarity modes (bipolar, excitatory, inhibitory)
    - Global gain values

    All matrix initialization is handled here, keeping ContinuousTimeRNN
    focused purely on the dynamical system dynamics.
    """

    def __init__(
        self,
        n_hidden: int = 64,
        n_inputs: int = 25,
        n_outputs: int = 32,
        base_seed: int = 42,
    ):
        self.n_hidden = n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.base_seed = base_seed

    # Matrix initialization methods
    def _initialize_integration_matrix(
        self,
        r_int: float,
        sparsity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Initialize W_int: sparse Gaussian with controlled spectral radius.

        Args:
            r_int: Target spectral radius (0.3-0.8)
            sparsity: Sparsity level [0, 1]
            rng: Random number generator

        Returns:
            W_int matrix (n_hidden x n_hidden)
        """
        # Create sparse mask
        mask = rng.random((self.n_hidden, self.n_hidden)) < sparsity

        # Initialize weights from normal distribution
        W = rng.normal(0, 1.0, (self.n_hidden, self.n_hidden))
        W = W * mask

        # Scale to target spectral radius
        if np.any(mask):
            eigvals = np.linalg.eigvals(W)
            current_radius = np.max(np.abs(eigvals))
            if current_radius > 0:
                W = W * (r_int / current_radius)

        return W

    def _initialize_propagation_matrix(
        self,
        r_prop: float,
        sparsity: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Initialize W_prop: sparse symmetric matrix with controlled radius.

        Args:
            r_prop: Target spectral radius (0.05-0.3)
            sparsity: Sparsity level [0, 1]
            rng: Random number generator

        Returns:
            W_prop matrix (n_hidden x n_hidden)
        """
        # Create sparse asymmetric matrix
        mask = rng.random((self.n_hidden, self.n_hidden)) < sparsity
        A = rng.normal(0, 1.0, (self.n_hidden, self.n_hidden))
        A = A * mask

        # Make symmetric: (A + A^T) / 2
        W = (A + A.T) / 2.0

        # Scale to target spectral radius
        if np.any(W != 0):
            eigvals = np.linalg.eigvals(W)
            current_radius = np.max(np.abs(eigvals))
            if current_radius > 0:
                W = W * (r_prop / current_radius)

        return W

    def _initialize_input_matrix(
        self,
        polarity: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Initialize W_u based on polarity mode.

        Args:
            polarity: "bipolar", "excitatory", or "inhibitory"
            rng: Random number generator

        Returns:
            W_u matrix (n_hidden x n_inputs)
        """
        if polarity == "excitatory":
            # Positive weights only
            W = rng.uniform(0, 1.0, (self.n_hidden, self.n_inputs))
        else:
            # Bipolar or inhibitory: Gaussian
            W = rng.normal(
                0, 1.0 / np.sqrt(self.n_inputs), (self.n_hidden, self.n_inputs)
            )
        return W

    def _initialize_output_matrix(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Initialize W_o: dense, small magnitude.

        Args:
            rng: Random number generator

        Returns:
            W_o matrix (n_outputs x n_hidden)
        """
        return rng.normal(
            0, 0.1 / np.sqrt(self.n_hidden), (self.n_outputs, self.n_hidden)
        )

    def generate_passive_system(
        self,
        polarity: str = "bipolar",
        g: float = 0.1,
        r_int: float = 0.5,
        r_prop: float = 0.1,
        V_rest: float = -65.0,
        tau_range: tuple = (10.0, 100.0),
        tau_distribution: str = "uniform",
        sparsity: float = 0.1,
        phi: str = "tanh",
        seed: Optional[int] = None,
    ) -> ContinuousTimeRNN:
        """Generate a passive subthreshold dynamical system.

        This is the main factory method for creating systems according to
        the design_2.2.md specification.

        Args:
            polarity: "bipolar", "excitatory", or "inhibitory"
            g: Global recurrent gain (default: 0.1)
            r_int: Spectral radius for W_int (default: 0.5, range: 0.3-0.8)
            r_prop: Spectral radius for W_prop (default: 0.1, range: 0.05-0.3)
            V_rest: Resting membrane potential in mV (default: -65.0)
            tau_range: (min, max) timescale in ms
            tau_distribution: "uniform" or "loguniform"
            sparsity: Sparsity for recurrent matrices (default: 0.1)
            phi: Nonlinearity - "tanh" or "softplus"
            seed: Random seed (auto-generated if None)

        Returns:
            Configured ContinuousTimeRNN instance
        """
        if seed is None:
            seed = self.base_seed + np.random.randint(0, 1000000)

        rng = np.random.default_rng(seed)

        # Generate weight matrices
        W_int = self._initialize_integration_matrix(r_int, sparsity, rng)
        W_prop = self._initialize_propagation_matrix(r_prop, sparsity, rng)
        W_u = self._initialize_input_matrix(polarity, rng)
        W_o = self._initialize_output_matrix(rng)

        # Generate timescales
        if tau_distribution == "loguniform":
            log_min = np.log(tau_range[0])
            log_max = np.log(tau_range[1])
            log_tau = rng.uniform(log_min, log_max, size=self.n_hidden)
            tau = np.exp(log_tau)
        else:
            tau = rng.uniform(tau_range[0], tau_range[1], size=self.n_hidden)

        return ContinuousTimeRNN(
            n_hidden=self.n_hidden,
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            W_int=W_int,
            W_prop=W_prop,
            W_u=W_u,
            W_o=W_o,
            g=g,
            tau=tau,
            V_rest=V_rest,
            polarity=polarity,
            phi=phi,
        )

    def generate_spectral_radius_sweep(
        self,
        radii: List[float],
        sparsity: float = 0.1,
        sweep_target: str = "r_int",
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with varying spectral radii.

        Args:
            radii: List of spectral radius values to test
            sparsity: Fixed sparsity level
            sweep_target: "r_int" or "r_prop" - which matrix to vary

        Returns:
            List of systems, one per radius value
        """
        systems = []
        for i, radius in enumerate(radii):
            kwargs = {
                "sparsity": sparsity,
                "seed": self.base_seed + i,
            }
            if sweep_target == "r_int":
                kwargs["r_int"] = radius
            else:
                kwargs["r_prop"] = radius

            system = self.generate_passive_system(**kwargs)
            systems.append(system)
        return systems

    def generate_sparsity_sweep(
        self,
        sparsities: List[float],
        r_int: float = 0.5,
        r_prop: float = 0.1,
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with varying sparsity levels.

        Args:
            sparsities: List of sparsity values [0, 1]
            r_int: Fixed integration matrix spectral radius
            r_prop: Fixed propagation matrix spectral radius

        Returns:
            List of systems, one per sparsity value
        """
        systems = []
        for i, sparsity in enumerate(sparsities):
            system = self.generate_passive_system(
                r_int=r_int,
                r_prop=r_prop,
                sparsity=sparsity,
                seed=self.base_seed + 1000 + i,
            )
            systems.append(system)
        return systems

    def generate_timescale_sweep(
        self,
        tau_ranges: List[tuple],
        r_int: float = 0.5,
        r_prop: float = 0.1,
        sparsity: float = 0.1,
        tau_distribution: str = "uniform",
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with varying timescale distributions.

        Args:
            tau_ranges: List of (min_tau, max_tau) tuples in ms
            r_int: Fixed integration matrix spectral radius
            r_prop: Fixed propagation matrix spectral radius
            sparsity: Fixed sparsity
            tau_distribution: "uniform" or "loguniform"

        Returns:
            List of systems, one per timescale range
        """
        systems = []
        for i, tau_range in enumerate(tau_ranges):
            system = self.generate_passive_system(
                r_int=r_int,
                r_prop=r_prop,
                sparsity=sparsity,
                tau_range=tau_range,
                tau_distribution=tau_distribution,
                seed=self.base_seed + 2000 + i,
            )
            systems.append(system)
        return systems

    def generate_polarity_comparison(
        self,
        base_params: Optional[dict] = None,
    ) -> dict:
        """Generate three systems with same parameters but different polarities.

        Args:
            base_params: Optional dict of parameters for generate_passive_system()

        Returns:
            Dict with keys "bipolar", "excitatory", "inhibitory"
        """
        if base_params is None:
            base_params = {}

        systems = {}
        for polarity in ["bipolar", "excitatory", "inhibitory"]:
            params = base_params.copy()
            params["polarity"] = polarity
            params["seed"] = self.base_seed + hash(polarity) % 1000
            systems[polarity] = self.generate_passive_system(**params)
        return systems

    def generate_gain_sweep(
        self,
        gain_values: List[float],
        r_int: float = 0.5,
        r_prop: float = 0.1,
        sparsity: float = 0.1,
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with varying global gain values.

        Args:
            gain_values: List of global gain g values
            r_int: Fixed integration matrix spectral radius
            r_prop: Fixed propagation matrix spectral radius
            sparsity: Fixed sparsity

        Returns:
            List of systems, one per gain value
        """
        systems = []
        for i, g in enumerate(gain_values):
            system = self.generate_passive_system(
                g=g,
                r_int=r_int,
                r_prop=r_prop,
                sparsity=sparsity,
                seed=self.base_seed + 4000 + i,
            )
            systems.append(system)
        return systems

    def generate_ensemble(
        self,
        n_systems: int,
        r_int_range: tuple = (0.3, 0.8),
        r_prop_range: tuple = (0.05, 0.3),
        g_range: tuple = (0.0, 0.3),
        sparsity_range: tuple = (0.05, 0.3),
    ) -> List[ContinuousTimeRNN]:
        """Generate diverse ensemble of random systems.

        Args:
            n_systems: Number of systems to generate
            r_int_range: (min, max) for integration matrix radius
            r_prop_range: (min, max) for propagation matrix radius
            g_range: (min, max) for global gain
            sparsity_range: (min, max) sparsity level

        Returns:
            List of randomly varied systems
        """
        systems = []
        for i in range(n_systems):
            r_int = np.random.uniform(*r_int_range)
            r_prop = np.random.uniform(*r_prop_range)
            g = np.random.uniform(*g_range)
            sparsity = np.random.uniform(*sparsity_range)
            system = self.generate_passive_system(
                r_int=r_int,
                r_prop=r_prop,
                g=g,
                sparsity=sparsity,
                seed=self.base_seed + 5000 + i,
            )
            systems.append(system)
        return systems

    # Operating regime presets (design_2.3.md)
    def generate_passive_mode_system(
        self,
        polarity: str = "bipolar",
        seed: Optional[int] = None,
    ) -> ContinuousTimeRNN:
        """Generate system in passive filtering mode.

        Leak-dominated dynamics for stable membrane-like filtering.
        Best for: initial validation, cable equation approximation

        Parameter regime (design_2.3.md Section 3):
        - g: 0 to 0.2 (weak recurrence)
        - r_prop: 0.05 to 0.2 (small spectral radius)
        - Behavior: EPSP/IPSP-like fluctuations, exponential decay

        Args:
            polarity: "bipolar", "excitatory", or "inhibitory"
            seed: Random seed

        Returns:
            CTRNN configured for passive filtering
        """
        return self.generate_passive_system(
            polarity=polarity,
            g=0.0,  # Start with zero recurrence as recommended
            r_int=0.5,
            r_prop=0.1,
            sparsity=0.1,
            seed=seed,
        )

    def generate_weakly_active_system(
        self,
        polarity: str = "bipolar",
        seed: Optional[int] = None,
    ) -> ContinuousTimeRNN:
        """Generate system in weakly active mode.

        Moderate recurrence for richer operator diversity.
        Best for: exploring functional differences between systems

        Parameter regime (design_2.3.md Section 4, Phase 2):
        - g: 0.2 to 0.4 (moderate recurrence)
        - r_prop: 0.2 to 0.5 (moderate spectral radius)
        - Behavior: longer memory, nonlinear amplification

        Args:
            polarity: "bipolar", "excitatory", or "inhibitory"
            seed: Random seed

        Returns:
            CTRNN configured for weakly active dynamics
        """
        return self.generate_passive_system(
            polarity=polarity,
            g=0.3,
            r_int=0.6,
            r_prop=0.3,
            sparsity=0.15,
            seed=seed,
        )

    def generate_active_mode_system(
        self,
        polarity: str = "bipolar",
        seed: Optional[int] = None,
    ) -> ContinuousTimeRNN:
        """Generate system in active nonlinear mode.

        Strong recurrence for rich dynamical computation.
        Best for: testing SI descriptor robustness, oscillatory systems

        Parameter regime (design_2.3.md Section 4, Phase 3):
        - g: 0.3 to 1.0 (strong recurrence)
        - r_prop: 0.5 to 1.0 (larger spectral radius)
        - Behavior: oscillatory transients, complex autocorrelation

        Args:
            polarity: "bipolar", "excitatory", or "inhibitory"
            seed: Random seed

        Returns:
            CTRNN configured for active nonlinear dynamics
        """
        return self.generate_passive_system(
            polarity=polarity,
            g=0.6,
            r_int=0.7,
            r_prop=0.6,
            sparsity=0.2,
            seed=seed,
        )

    def generate_regime_comparison(
        self,
        polarity: str = "bipolar",
    ) -> dict:
        """Generate three systems spanning passive to active regimes.

        Creates systems with identical dimensions but different dynamical
        regimes for comparative analysis.

        Args:
            polarity: Polarity mode for all systems

        Returns:
            Dict with keys "passive", "weakly_active", "active"
        """
        return {
            "passive": self.generate_passive_mode_system(polarity, seed=1000),
            "weakly_active": self.generate_weakly_active_system(polarity, seed=2000),
            "active": self.generate_active_mode_system(polarity, seed=3000),
        }

    def generate_activity_scale_sweep(
        self,
        n_systems: int = 5,
        polarity: str = "bipolar",
    ) -> List[ContinuousTimeRNN]:
        """Generate systems with gradually increasing activity levels.

        Sweeps global gain g from passive (g=0) to weakly active (g=0.4)
        while keeping other parameters stable.

        Args:
            n_systems: Number of systems to generate
            polarity: Polarity mode

        Returns:
            List of systems with increasing activity
        """
        systems = []
        for i in range(n_systems):
            # Linear interpolation from g=0.0 to g=0.4
            g = 0.0 + (0.4 * i / (n_systems - 1)) if n_systems > 1 else 0.0
            system = self.generate_passive_system(
                polarity=polarity,
                g=g,
                r_int=0.5,
                r_prop=0.1 + (0.2 * i / (n_systems - 1)),
                sparsity=0.1,
                seed=self.base_seed + 6000 + i,
            )
            systems.append(system)
        return systems

    # Backward compatibility alias
    def generate_random_system(
        self,
        spectral_radius: float = 0.5,
        sparsity: float = 0.1,
        tau_range: tuple = (10.0, 100.0),
        seed: Optional[int] = None,
    ) -> ContinuousTimeRNN:
        """Legacy method - now calls generate_passive_system with r_int.

        Args:
            spectral_radius: Mapped to r_int for backward compatibility
            sparsity: Sparsity level
            tau_range: (min, max) timescale
            seed: Random seed

        Returns:
            Configured ContinuousTimeRNN instance
        """
        return self.generate_passive_system(
            r_int=spectral_radius,
            sparsity=sparsity,
            tau_range=tau_range,
            seed=seed,
        )
