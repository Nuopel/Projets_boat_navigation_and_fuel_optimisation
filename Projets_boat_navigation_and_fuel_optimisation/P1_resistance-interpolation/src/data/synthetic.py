"""Synthetic data generation for interpolation validation.

This module provides the SyntheticSurfaceGenerator class for creating
controlled resistance surfaces with known ground truth, enabling objective
validation of interpolation methods.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.stats import qmc


class SyntheticSurfaceGenerator:
    """Generate synthetic ship resistance surfaces with known ground truth.

    Creates physically realistic resistance surfaces using a parametric formula,
    with controllable noise levels and sampling strategies. Enables objective
    evaluation of interpolation methods since the true surface is known.

    Attributes:
        v_range: (min, max) velocity bounds in knots
        t_range: (min, max) draft bounds in meters
        noise_std: Standard deviation of Gaussian noise to add
        random_state: Random seed for reproducibility

    Example:
        >>> generator = SyntheticSurfaceGenerator()
        >>> ground_truth = generator.create_dense_grid(grid_size=100)
        >>> sparse_data = generator.sample_sparse(n_samples=30, strategy='latin_hypercube')
    """

    def __init__(
        self,
        v_range: Tuple[float, float] = (10.0, 25.0),
        t_range: Tuple[float, float] = (6.0, 10.0),
        noise_std: float = 0.01,
        random_state: int = 42
    ):
        """Initialize the synthetic surface generator.

        Args:
            v_range: (min, max) velocity bounds in knots
            t_range: (min, max) draft bounds in meters
            noise_std: Standard deviation of additive Gaussian noise
            random_state: Random seed for reproducibility
        """
        self.v_range = v_range
        self.t_range = t_range
        self.noise_std = noise_std
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def resistance_function(self, V: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Generate true resistance surface using physically inspired formula.

        The resistance model mimics typical yacht/ship behavior:
        R(V, T) = a*V² + b/T + c*V*T + d

        Physical interpretation:
        - V² term: Velocity-dependent resistance (dominates at high speed)
        - 1/T term: Draft influences wetted surface area
        - V*T term: Interaction effect between speed and draft
        - Constant: Baseline resistance

        Coefficients chosen to produce realistic resistance values in range [0, 50].

        Args:
            V: Velocity array in knots
            T: Draft array in meters

        Returns:
            Resistance array (dimensionless)

        Example:
            >>> gen = SyntheticSurfaceGenerator()
            >>> V, T = np.meshgrid(np.linspace(10, 25, 100), np.linspace(6, 10, 100))
            >>> R = gen.resistance_function(V.ravel(), T.ravel())
        """
        # Coefficients for physically realistic surface
        a = 0.05   # Velocity-squared coefficient (strong effect)
        b = -2.0   # Inverse draft coefficient (negative: larger draft → less resistance)
        c = 0.01   # Interaction coefficient
        d = 15.0   # Baseline resistance

        # Compute resistance
        R = a * V**2 + b / T + c * V * T + d

        return R

    def create_dense_grid(self, grid_size: int = 100) -> Dict[str, np.ndarray]:
        """Generate ground truth resistance on a fine regular grid.

        Creates a dense grid for computing exact interpolation errors and
        visualizing the true surface.

        Args:
            grid_size: Number of points along each dimension (total points = grid_size²)

        Returns:
            Dictionary containing:
            - 'V': Velocity array (grid_size²,)
            - 'T': Draft array (grid_size²,)
            - 'R': Resistance array (grid_size²,)
            - 'V_grid': 2D velocity grid (grid_size, grid_size)
            - 'T_grid': 2D draft grid (grid_size, grid_size)
            - 'R_grid': 2D resistance grid (grid_size, grid_size)

        Example:
            >>> gen = SyntheticSurfaceGenerator()
            >>> grid = gen.create_dense_grid(grid_size=100)
            >>> print(f"Total grid points: {len(grid['V'])}")  # 10,000
        """
        # Create 1D arrays for each dimension
        v_vals = np.linspace(self.v_range[0], self.v_range[1], grid_size)
        t_vals = np.linspace(self.t_range[0], self.t_range[1], grid_size)

        # Create 2D meshgrid
        V_grid, T_grid = np.meshgrid(v_vals, t_vals)

        # Compute resistance on grid (no noise for ground truth)
        R_grid = self.resistance_function(V_grid, T_grid)

        # Flatten for 1D arrays
        V_flat = V_grid.ravel()
        T_flat = T_grid.ravel()
        R_flat = R_grid.ravel()

        return {
            'V': V_flat,
            'T': T_flat,
            'R': R_flat,
            'V_grid': V_grid,
            'T_grid': T_grid,
            'R_grid': R_grid
        }

    def sample_sparse(
        self,
        n_samples: int,
        strategy: str = 'random',
        add_noise: bool = True,
        snr_db: float = 20.0
    ) -> pd.DataFrame:
        """Generate sparse training samples using various sampling strategies.

        Args:
            n_samples: Number of samples to generate
            strategy: Sampling strategy - 'random', 'latin_hypercube', or 'structured'
            add_noise: Whether to add Gaussian noise to resistance values
            snr_db: Signal-to-noise ratio in dB (higher = less noise)

        Returns:
            DataFrame with columns ['V', 'T', 'R']

        Raises:
            ValueError: If strategy is not recognized

        Example:
            >>> gen = SyntheticSurfaceGenerator()
            >>> sparse_data = gen.sample_sparse(30, strategy='latin_hypercube', snr_db=20)
            >>> print(sparse_data.shape)  # (30, 3)
        """
        if strategy == 'random':
            # Pure random sampling
            V_samples = self.rng.uniform(
                self.v_range[0], self.v_range[1], size=n_samples
            )
            T_samples = self.rng.uniform(
                self.t_range[0], self.t_range[1], size=n_samples
            )

        elif strategy == 'latin_hypercube':
            # Latin Hypercube Sampling for better space-filling
            sampler = qmc.LatinHypercube(d=2, seed=self.random_state)
            samples = sampler.random(n=n_samples)

            # Scale to actual ranges
            l_bounds = [self.v_range[0], self.t_range[0]]
            u_bounds = [self.v_range[1], self.t_range[1]]
            scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

            V_samples = scaled_samples[:, 0]
            T_samples = scaled_samples[:, 1]

        elif strategy == 'structured':
            # Structured grid with some jitter
            grid_1d = int(np.ceil(np.sqrt(n_samples)))
            v_vals = np.linspace(self.v_range[0], self.v_range[1], grid_1d)
            t_vals = np.linspace(self.t_range[0], self.t_range[1], grid_1d)
            V_grid, T_grid = np.meshgrid(v_vals, t_vals)

            V_samples = V_grid.ravel()[:n_samples]
            T_samples = T_grid.ravel()[:n_samples]

            # Add small jitter (5% of range)
            v_jitter = (self.v_range[1] - self.v_range[0]) * 0.05
            t_jitter = (self.t_range[1] - self.t_range[0]) * 0.05

            V_samples += self.rng.uniform(-v_jitter, v_jitter, size=len(V_samples))
            T_samples += self.rng.uniform(-t_jitter, t_jitter, size=len(T_samples))

            # Clip to bounds
            V_samples = np.clip(V_samples, self.v_range[0], self.v_range[1])
            T_samples = np.clip(T_samples, self.t_range[0], self.t_range[1])

        else:
            raise ValueError(
                f"Unknown sampling strategy '{strategy}'. "
                "Choose from: 'random', 'latin_hypercube', 'structured'"
            )

        # Compute true resistance
        R_clean = self.resistance_function(V_samples, T_samples)

        # Add noise if requested
        if add_noise:
            R_samples = self.add_noise(R_clean, snr_db=snr_db)
        else:
            R_samples = R_clean

        # Create DataFrame
        df = pd.DataFrame({
            'V': V_samples,
            'T': T_samples,
            'R': R_samples
        })

        return df

    def add_noise(self, R: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add Gaussian noise with specified Signal-to-Noise Ratio.

        SNR (dB) = 10 * log10(signal_power / noise_power)

        Args:
            R: Clean resistance signal
            snr_db: Target signal-to-noise ratio in decibels

        Returns:
            Noisy resistance signal

        Example:
            >>> gen = SyntheticSurfaceGenerator()
            >>> R_clean = np.array([10.0, 20.0, 30.0])
            >>> R_noisy = gen.add_noise(R_clean, snr_db=20)
        """
        # Calculate signal power
        signal_power = np.mean(R ** 2)

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)

        # Calculate required noise power
        noise_power = signal_power / snr_linear

        # Generate noise with appropriate standard deviation
        noise_std = np.sqrt(noise_power)
        noise = self.rng.normal(0, noise_std, size=R.shape)

        # Add noise to signal
        R_noisy = R + noise

        return R_noisy

    def get_domain_info(self) -> Dict[str, float]:
        """Get information about the domain bounds.

        Returns:
            Dictionary with domain parameters:
            - V_min, V_max: Velocity bounds
            - T_min, T_max: Draft bounds
            - V_range, T_range: Ranges (max - min)
            - area: Domain area
        """
        return {
            'V_min': self.v_range[0],
            'V_max': self.v_range[1],
            'T_min': self.t_range[0],
            'T_max': self.t_range[1],
            'V_range': self.v_range[1] - self.v_range[0],
            'T_range': self.t_range[1] - self.t_range[0],
            'area': (self.v_range[1] - self.v_range[0]) * (self.t_range[1] - self.t_range[0])
        }

    def compute_snr(self, R_clean: np.ndarray, R_noisy: np.ndarray) -> float:
        """Compute actual SNR between clean and noisy signals.

        Useful for validating that add_noise() produces expected SNR.

        Args:
            R_clean: Clean resistance signal
            R_noisy: Noisy resistance signal

        Returns:
            SNR in decibels

        Example:
            >>> gen = SyntheticSurfaceGenerator()
            >>> R_clean = np.array([10, 20, 30])
            >>> R_noisy = gen.add_noise(R_clean, snr_db=20)
            >>> actual_snr = gen.compute_snr(R_clean, R_noisy)
            >>> print(f"Target: 20 dB, Actual: {actual_snr:.1f} dB")
        """
        signal_power = np.mean(R_clean ** 2)
        noise = R_noisy - R_clean
        noise_power = np.mean(noise ** 2)

        if noise_power == 0:
            return np.inf

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

        return snr_db
