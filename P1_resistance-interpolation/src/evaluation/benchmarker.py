"""Comprehensive benchmarking suite for interpolation methods.

This module provides systematic evaluation of RBF, Spline, and Kriging interpolators
on both synthetic and real yacht hydrodynamics data.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler

from src.interpolators import RBFInterpolator, SplineInterpolator, KrigingInterpolator
from src.evaluation.metrics import MetricsCalculator


@dataclass
class BenchmarkResult:
    """Container for benchmark results from a single experiment.

    Attributes:
        method_name: Name of interpolation method
        n_train: Number of training samples
        n_test: Number of test samples
        rmse: Root mean squared error
        mae: Mean absolute error
        r2: R-squared score
        max_error: Maximum absolute error
        train_time: Training time in seconds
        predict_time: Prediction time in seconds
        config: Method-specific configuration
        noise_level: SNR if noise was added (None otherwise)
    """
    method_name: str
    n_train: int
    n_test: int
    rmse: float
    mae: float
    r2: float
    max_error: float
    train_time: float
    predict_time: float
    config: Dict = field(default_factory=dict)
    noise_level: Optional[float] = None
    normalized: bool = False

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            'method': self.method_name,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'max_error': self.max_error,
            'train_time': self.train_time,
            'predict_time': self.predict_time,
            'noise_snr': self.noise_level,
            'normalized': self.normalized,
            **{f'config_{k}': v for k, v in self.config.items()}
        }


class InterpolationBenchmarker:
    """Comprehensive benchmarking for interpolation methods.

    This class provides systematic evaluation across multiple dimensions:
    - Different training set sizes (convergence analysis)
    - Different noise levels (robustness testing)
    - Multiple method configurations
    - Statistical significance testing

    Example:
        >>> from src.evaluation.benchmarker import InterpolationBenchmarker
        >>> benchmarker = InterpolationBenchmarker()
        >>>
        >>> # Load data
        >>> benchmarker.load_yacht_data('data/yacht_hydro.xls')
        >>>
        >>> # Run convergence analysis
        >>> results = benchmarker.run_convergence_study(
        ...     sample_sizes=[10, 20, 50, 100, 200],
        ...     n_trials=5
        ... )
        >>>
        >>> # Get summary
        >>> summary = benchmarker.get_summary_statistics(results)
    """

    def __init__(self, random_state: int = 42):
        """Initialize benchmarker.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Data storage
        self.X_full = None
        self.y_full = None
        self.data_source = None

        # Results storage
        self.results: List[BenchmarkResult] = []

    def load_yacht_data(
        self,
        filepath: str,
        aggregate_duplicates: bool = False,
        agg_func: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load UCI Yacht Hydrodynamics data.

        Args:
            filepath: Path to yacht_hydro.xls file
            aggregate_duplicates: If True, aggregate duplicate (V, T) pairs.
            agg_func: Aggregation function name for duplicate (V, T) pairs.

        Returns:
            X: Features (V, T) array of shape (n_samples, 2)
            y: Resistance values array of shape (n_samples,)
        """
        from src.data.loader import YachtDataLoader
        from src.data.preprocessor import DataPreprocessor

        # Load and preprocess
        loader = YachtDataLoader(filepath)
        df = loader.load()

        preprocessor = DataPreprocessor()
        df_vt = preprocessor.create_vt_surface(
            df,
            aggregate_duplicates=aggregate_duplicates,
            agg_func=agg_func
        )

        # Extract features and target
        self.X_full = df_vt[['V', 'T']].values
        self.y_full = df_vt['R'].values
        self.data_source = 'yacht_uci'

        return self.X_full, self.y_full

    def load_synthetic_data(
        self,
        n_samples: int = 200,
        strategy: str = 'latin_hypercube',
        add_noise: bool = False,
        noise_snr: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load synthetic resistance data.

        Args:
            n_samples: Number of samples to generate
            strategy: Sampling strategy ('random', 'latin_hypercube', 'structured')
            add_noise: Whether to add noise
            noise_snr: Signal-to-noise ratio in dB (if add_noise=True)

        Returns:
            X: Features (V, T) array
            y: Resistance values array
        """
        from src.data.synthetic import SyntheticSurfaceGenerator

        generator = SyntheticSurfaceGenerator(random_state=self.random_state)
        df = generator.sample_sparse(
            n_samples=n_samples,
            strategy=strategy,
            add_noise=add_noise,
            snr_db=noise_snr
        )

        self.X_full = df[['V', 'T']].values
        self.y_full = df['R'].values
        self.data_source = 'synthetic'

        return self.X_full, self.y_full

    def benchmark_single_method(
        self,
        method_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        config: Optional[Dict] = None,
        noise_level: Optional[float] = None,
        normalize: bool = False
    ) -> BenchmarkResult:
        """Benchmark a single interpolation method.

        Args:
            method_name: 'rbf', 'spline', or 'kriging'
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets (ground truth)
            config: Method-specific configuration
            noise_level: SNR if noise was added (for recording)
            normalize: If True, normalize features using StandardScaler

        Returns:
            BenchmarkResult containing all metrics and timings
        """
        if config is None:
            config = {}

        # Normalize features if requested
        if normalize:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            X_test_scaled = scaler_X.transform(X_test)
        else:
            X_train_scaled = X_train
            y_train_scaled = y_train
            X_test_scaled = X_test

        # Create interpolator
        if method_name.lower() == 'rbf':
            interpolator = RBFInterpolator(
                kernel=config.get('kernel', 'thin_plate_spline'),
                smoothing=config.get('smoothing', 0.0),
                epsilon=config.get('epsilon', None)
            )
        elif method_name.lower() == 'spline':
            interpolator = SplineInterpolator(
                kx=config.get('kx', 3),
                ky=config.get('ky', 3),
                smoothing=config.get('smoothing', 0.0)
            )
        elif method_name.lower() == 'kriging':
            interpolator = KrigingInterpolator(
                kernel_type=config.get('kernel_type', 'rbf'),
                alpha=config.get('alpha', 1e-6),
                n_restarts_optimizer=config.get('n_restarts_optimizer', 5)
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Train
        start_train = time.time()
        interpolator.fit(X_train_scaled, y_train_scaled)
        train_time = time.time() - start_train

        # Predict
        start_predict = time.time()
        y_pred_scaled = interpolator.predict(X_test_scaled)
        predict_time = time.time() - start_predict

        # Inverse transform predictions if normalized
        if normalize:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled

        # Compute metrics (on original scale)
        metrics = MetricsCalculator.compute_all_metrics(y_test, y_pred)

        # Create result
        result = BenchmarkResult(
            method_name=method_name,
            n_train=len(X_train),
            n_test=len(X_test),
            rmse=metrics['rmse'],
            mae=metrics['mae'],
            r2=metrics['r2'],
            max_error=metrics['max_error'],
            train_time=train_time,
            predict_time=predict_time,
            config=config,
            noise_level=noise_level,
            normalized=normalize
        )

        return result

    def run_convergence_study(
        self,
        sample_sizes: List[int],
        methods: Optional[List[str]] = None,
        n_trials: int = 5,
        test_fraction: float = 0.3,
        normalize: bool = False
    ) -> List[BenchmarkResult]:
        """Run convergence analysis with varying training set sizes.

        Tests how accuracy improves as training data increases.

        Args:
            sample_sizes: List of training set sizes to test
            methods: List of methods to test (default: all three)
            n_trials: Number of random trials per configuration
            test_fraction: Fraction of data to use for testing
            normalize: If True, normalize features before training

        Returns:
            List of BenchmarkResult objects
        """
        if self.X_full is None:
            raise ValueError("No data loaded. Call load_yacht_data() or load_synthetic_data() first.")

        if methods is None:
            methods = ['rbf', 'spline', 'kriging']

        results = []

        # Define default configurations for each method
        configs = {
            'rbf': {'kernel': 'thin_plate_spline', 'smoothing': 0.0},
            'spline': {'kx': 3, 'ky': 3, 'smoothing': 0.0},
            'kriging': {'kernel_type': 'rbf', 'alpha': 1e-6, 'n_restarts_optimizer': 5}
        }

        for n_train in sample_sizes:
            # Check if we have enough data
            n_total_needed = int(n_train / (1 - test_fraction))
            if n_total_needed > len(self.X_full):
                print(f"Warning: Skipping n_train={n_train}, need {n_total_needed} samples but only have {len(self.X_full)}")
                continue

            for trial in range(n_trials):
                # Random train/test split
                indices = self.rng.permutation(len(self.X_full))[:n_total_needed]
                n_test = n_total_needed - n_train

                train_idx = indices[:n_train]
                test_idx = indices[n_train:n_total_needed]

                X_train = self.X_full[train_idx]
                y_train = self.y_full[train_idx]
                X_test = self.X_full[test_idx]
                y_test = self.y_full[test_idx]

                # Test each method
                for method in methods:
                    try:
                        result = self.benchmark_single_method(
                            method_name=method,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            config=configs[method],
                            normalize=normalize
                        )
                        results.append(result)

                        norm_str = " [normalized]" if normalize else ""
                        print(f"✓ {method.upper()} | n_train={n_train} | trial={trial+1}/{n_trials} | RMSE={result.rmse:.4f}{norm_str}")

                    except Exception as e:
                        print(f"✗ {method.upper()} | n_train={n_train} | trial={trial+1}/{n_trials} | Error: {str(e)}")

        self.results.extend(results)
        return results

    def run_noise_robustness_study(
        self,
        noise_levels: List[float],
        n_train: int = 100,
        methods: Optional[List[str]] = None,
        n_trials: int = 5,
        normalize: bool = False
    ) -> List[BenchmarkResult]:
        """Test robustness to measurement noise.

        Args:
            noise_levels: List of SNR values in dB (higher = less noise)
            n_train: Training set size
            methods: Methods to test
            n_trials: Trials per configuration
            normalize: If True, normalize features before training

        Returns:
            List of BenchmarkResult objects
        """
        if self.X_full is None:
            raise ValueError("No data loaded.")

        if methods is None:
            methods = ['rbf', 'spline', 'kriging']

        results = []

        # Configurations with smoothing for noisy data
        configs_clean = {
            'rbf': {'kernel': 'thin_plate_spline', 'smoothing': 0.0},
            'spline': {'kx': 3, 'ky': 3, 'smoothing': 0.0},
            'kriging': {'kernel_type': 'rbf', 'alpha': 1e-6, 'n_restarts_optimizer': 3}
        }

        configs_noisy = {
            'rbf': {'kernel': 'thin_plate_spline', 'smoothing': 0.1},
            'spline': {'kx': 3, 'ky': 3, 'smoothing': 1.0},
            'kriging': {'kernel_type': 'rbf', 'alpha': 1e-3, 'n_restarts_optimizer': 3}
        }

        for snr_db in noise_levels:
            for trial in range(n_trials):
                # Split data
                indices = self.rng.permutation(len(self.X_full))
                train_idx = indices[:n_train]
                test_idx = indices[n_train:n_train+50]  # Fixed test size

                X_train = self.X_full[train_idx]
                y_train_clean = self.y_full[train_idx]
                X_test = self.X_full[test_idx]
                y_test = self.y_full[test_idx]

                # Add noise to training data
                signal_power = np.var(y_train_clean)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = self.rng.randn(len(y_train_clean)) * np.sqrt(noise_power)
                y_train_noisy = y_train_clean + noise

                # Test each method
                for method in methods:
                    # Choose config based on noise level
                    config = configs_noisy if snr_db < 20 else configs_clean

                    try:
                        result = self.benchmark_single_method(
                            method_name=method,
                            X_train=X_train,
                            y_train=y_train_noisy,
                            X_test=X_test,
                            y_test=y_test,
                            config=config[method],
                            noise_level=snr_db,
                            normalize=normalize
                        )
                        results.append(result)

                        norm_str = " [normalized]" if normalize else ""
                        print(f"✓ {method.upper()} | SNR={snr_db}dB | trial={trial+1} | RMSE={result.rmse:.4f}{norm_str}")

                    except Exception as e:
                        print(f"✗ {method.upper()} | SNR={snr_db}dB | trial={trial+1} | Error: {str(e)}")

        self.results.extend(results)
        return results

    def get_summary_statistics(
        self,
        results: Optional[List[BenchmarkResult]] = None
    ) -> pd.DataFrame:
        """Compute summary statistics from benchmark results.

        Args:
            results: List of results (uses self.results if None)

        Returns:
            DataFrame with mean and std for each method/configuration
        """
        if results is None:
            results = self.results

        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])

        # Group by method and sample size
        if 'n_train' in df.columns:
            groupby_cols = ['method', 'n_train']
        else:
            groupby_cols = ['method']

        # Compute statistics
        summary = df.groupby(groupby_cols).agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'train_time': ['mean', 'std'],
            'predict_time': ['mean', 'std']
        }).reset_index()

        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

        return summary

    def export_results(self, filepath: str):
        """Export all results to CSV.

        Args:
            filepath: Output CSV path
        """
        if not self.results:
            print("No results to export.")
            return

        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(filepath, index=False)
        print(f"Exported {len(self.results)} results to {filepath}")

    def clear_results(self):
        """Clear all stored results."""
        self.results = []
