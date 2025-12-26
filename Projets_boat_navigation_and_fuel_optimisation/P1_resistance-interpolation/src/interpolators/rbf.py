"""Radial Basis Function (RBF) interpolation for ship performance prediction.

This module provides RBF interpolation using scipy's RBFInterpolator,
which is excellent for irregularly spaced data points.
"""

import time
import numpy as np
from scipy.interpolate import RBFInterpolator as ScipyRBFInterpolator
from .base import BaseInterpolator


class RBFInterpolator(BaseInterpolator):
    """Radial Basis Function interpolation for 2D resistance surfaces.

    RBF interpolation works by fitting a linear combination of radial basis functions
    centered at each training point. Excellent for irregular, scattered data.

    Advantages:
    - No grid requirement (works with scattered points)
    - Flexible kernels (thin_plate_spline, multiquadric, gaussian, etc.)
    - Fast prediction after training
    - Handles irregular sampling well

    Limitations:
    - Can overfit without proper smoothing
    - Kernel choice affects results
    - Memory scales with O(n²) for training

    Attributes:
        kernel: RBF kernel function ('thin_plate_spline', 'multiquadric', 'gaussian', etc.)
        smoothing: Smoothing parameter (0 = exact interpolation, >0 = smoothed fit)
        epsilon: Shape parameter for some kernels (auto-selected if None)

    Example:
        >>> from src.interpolators.rbf import RBFInterpolator
        >>> interpolator = RBFInterpolator(kernel='thin_plate_spline')
        >>> interpolator.fit(X_train, y_train)
        >>> predictions = interpolator.predict(X_test)
    """

    def __init__(
        self,
        kernel: str = 'thin_plate_spline',
        smoothing: float = 0.0,
        epsilon: float = None,
        degree: int = None
    ):
        """Initialize RBF interpolator.

        Args:
            kernel: RBF kernel type. Options:
                - 'thin_plate_spline' (default): C² continuous, no free parameters
                - 'multiquadric': sqrt(1 + (epsilon*r)²)
                - 'inverse_multiquadric': 1/sqrt(1 + (epsilon*r)²)
                - 'gaussian': exp(-(epsilon*r)²)
                - 'linear': r
                - 'cubic': r³
                - 'quintic': r⁵
            smoothing: Smoothing parameter
                - 0.0 (default): Exact interpolation (passes through all points)
                - >0: Smoothed fit (reduces overfitting, useful with noisy data)
            epsilon: Shape parameter for kernels that use it
                - None (default): Auto-selected by scipy
                - >0: Manual shape parameter
            degree: Degree of the added polynomial term
                - None (default): SciPy default for each kernel
                - 0: No polynomial term (more robust for collinear points)
                - 1: Linear term (default for thin_plate_spline in SciPy)

        Example:
            >>> # Exact interpolation with thin-plate splines
            >>> rbf = RBFInterpolator(kernel='thin_plate_spline', smoothing=0.0)
            >>>
            >>> # Smoothed interpolation for noisy data
            >>> rbf_smooth = RBFInterpolator(kernel='thin_plate_spline', smoothing=0.01)
            >>>
            >>> # Gaussian RBF with manual epsilon
            >>> rbf_gaussian = RBFInterpolator(kernel='gaussian', epsilon=0.5)
        """
        super().__init__(name='RBF')
        self.kernel = kernel
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.degree = degree
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RBFInterpolator':
        """Train RBF interpolator on (V, T, R) data.

        Args:
            X: Training features (n_samples, 2) with columns [V, T]
            y: Training targets (n_samples,) resistance values

        Returns:
            self: Fitted interpolator instance

        Raises:
            ValueError: If X or y have invalid shapes or values

        Example:
            >>> X_train = np.array([[10, 6], [15, 7], [20, 8]])
            >>> y_train = np.array([20.0, 25.0, 35.0])
            >>> rbf = RBFInterpolator()
            >>> rbf.fit(X_train, y_train)
            RBF(fitted, n_samples=3, train_time=0.002s)
        """
        # Validate input data
        self._validate_training_data(X, y)

        # Record training start time
        start_time = time.time()

        # Build RBF interpolator
        rbf_kwargs = {
            'kernel': self.kernel,
            'smoothing': self.smoothing
        }

        if self.degree is not None:
            rbf_kwargs['degree'] = self.degree

        # Some kernels require epsilon parameter
        # Scale-invariant kernels: thin_plate_spline, linear, cubic, quintic
        # Non-scale-invariant kernels: gaussian, multiquadric, inverse_multiquadric
        scale_invariant_kernels = {'thin_plate_spline', 'linear', 'cubic', 'quintic'}

        if self.epsilon is not None:
            rbf_kwargs['epsilon'] = self.epsilon
        elif self.kernel not in scale_invariant_kernels:
            # Auto-set epsilon for non-scale-invariant kernels
            # Use a reasonable default based on data scale
            data_range = np.ptp(X, axis=0).mean()
            rbf_kwargs['epsilon'] = 1.0 / data_range if data_range > 0 else 1.0

        try:
            self._model = ScipyRBFInterpolator(X, y, **rbf_kwargs)
        except np.linalg.LinAlgError as exc:
            # Retry without polynomial term when points are collinear.
            if self.degree is None:
                rbf_kwargs['degree'] = 0
                self._model = ScipyRBFInterpolator(X, y, **rbf_kwargs)
            else:
                raise exc

        # Record training completion
        self.train_time = time.time() - start_time
        self.is_fitted = True
        self.n_training_samples = len(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict resistance at query points.

        Args:
            X: Query features (n_queries, 2) with columns [V, T]

        Returns:
            predictions: (n_queries,) predicted resistance values

        Raises:
            ValueError: If called before fit() or X has invalid shape

        Example:
            >>> X_test = np.array([[12, 6.5], [18, 7.5]])
            >>> predictions = rbf.predict(X_test)
            >>> print(predictions)
            [22.5, 30.2]
        """
        # Check if fitted
        self._check_is_fitted()

        # Validate input shape
        self._validate_input_shape(X)

        # Record prediction start time
        start_time = time.time()

        # Make predictions
        predictions = self._model(X)

        # Record prediction time
        self.predict_time = time.time() - start_time

        return predictions

    def get_metadata(self) -> dict:
        """Get metadata including RBF-specific parameters.

        Returns:
            Dictionary with base metadata plus:
            - kernel: RBF kernel function
            - smoothing: Smoothing parameter
            - epsilon: Shape parameter (if set)

        Example:
            >>> meta = rbf.get_metadata()
            >>> print(f"Kernel: {meta['kernel']}")
            >>> print(f"Training time: {meta['train_time']:.4f}s")
        """
        metadata = super().get_metadata()
        metadata.update({
            'kernel': self.kernel,
            'smoothing': self.smoothing,
            'epsilon': self.epsilon,
            'degree': self.degree
        })
        return metadata

    def __repr__(self) -> str:
        """String representation with RBF-specific info."""
        if self.is_fitted:
            return (
                f"RBF(kernel='{self.kernel}', smoothing={self.smoothing}, "
                f"fitted on {self.n_training_samples} samples, "
                f"train_time={self.train_time:.3f}s)"
            )
        else:
            return f"RBF(kernel='{self.kernel}', smoothing={self.smoothing}, not fitted)"
