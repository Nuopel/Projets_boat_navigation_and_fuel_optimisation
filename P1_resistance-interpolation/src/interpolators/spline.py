"""Bivariate Spline interpolation for ship performance prediction.

This module provides smooth bivariate spline interpolation using scipy's
SmoothBivariateSpline, which is fast and well-suited for gridded or semi-regular data.
"""

import time
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
from .base import BaseInterpolator


class SplineInterpolator(BaseInterpolator):
    """Bivariate spline interpolation for 2D resistance surfaces.

    Spline interpolation uses piecewise polynomial functions with smoothness
    constraints. Very fast for prediction and well-established mathematically.

    Advantages:
    - Very fast computation (both training and prediction)
    - Smoothing parameter controls overfitting
    - Mathematically well-founded
    - Stable numerical behavior

    Limitations:
    - Prefers semi-regular grids (works with irregular but less optimal)
    - May struggle with very sparse, scattered data
    - Less flexible than RBF for irregular geometries

    Attributes:
        smoothing: Smoothing factor s
            - s=0: Interpolating spline (passes through points if possible)
            - s>0: Smoothing spline (balances fit vs smoothness)
        kx, ky: Degrees of spline in x and y directions
            - 1: Linear
            - 3: Cubic (default, recommended)
            - 5: Quintic

    Example:
        >>> from src.interpolators.spline import SplineInterpolator
        >>> interpolator = SplineInterpolator(smoothing=0.0, kx=3, ky=3)
        >>> interpolator.fit(X_train, y_train)
        >>> predictions = interpolator.predict(X_test)
    """

    def __init__(
        self,
        smoothing: float = 0.0,
        kx: int = 3,
        ky: int = 3
    ):
        """Initialize Spline interpolator.

        Args:
            smoothing: Smoothing factor
                - 0.0 (default): Interpolating spline (exact fit)
                - >0: Smoothing spline (reduces overfitting)
                - Larger values → smoother surface, worse fit
            kx: Degree of spline in x-direction (velocity)
                - 1: Linear
                - 3: Cubic (default)
                - 5: Quintic
                Must satisfy: 1 <= kx <= 5
            ky: Degree of spline in y-direction (draft)
                - 1: Linear
                - 3: Cubic (default)
                - 5: Quintic
                Must satisfy: 1 <= ky <= 5

        Example:
            >>> # Cubic interpolating spline (default)
            >>> spline = SplineInterpolator()
            >>>
            >>> # Linear smoothing spline
            >>> spline_linear = SplineInterpolator(smoothing=0.1, kx=1, ky=1)
            >>>
            >>> # Quintic spline for very smooth surfaces
            >>> spline_smooth = SplineInterpolator(kx=5, ky=5)
        """
        super().__init__(name='Spline')
        self.smoothing = smoothing
        self.kx = kx
        self.ky = ky
        self._model = None

        # Validate spline degrees
        if not (1 <= kx <= 5):
            raise ValueError(f"kx must be between 1 and 5, got {kx}")
        if not (1 <= ky <= 5):
            raise ValueError(f"ky must be between 1 and 5, got {ky}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SplineInterpolator':
        """Train spline interpolator on (V, T, R) data.

        Args:
            X: Training features (n_samples, 2) with columns [V, T]
            y: Training targets (n_samples,) resistance values

        Returns:
            self: Fitted interpolator instance

        Raises:
            ValueError: If X or y have invalid shapes or values
            RuntimeError: If spline fitting fails (e.g., too few points)

        Example:
            >>> X_train = np.array([[10, 6], [15, 7], [20, 8]])
            >>> y_train = np.array([20.0, 25.0, 35.0])
            >>> spline = SplineInterpolator()
            >>> spline.fit(X_train, y_train)
            Spline(fitted, n_samples=3, train_time=0.001s)
        """
        # Validate input data
        self._validate_training_data(X, y)

        # Check minimum samples for spline degree
        # SmoothBivariateSpline requires (kx+1) * (ky+1) samples
        min_samples = (self.kx + 1) * (self.ky + 1)
        if len(X) < min_samples:
            raise ValueError(
                f"Need at least {min_samples} samples for spline degrees "
                f"kx={self.kx}, ky={self.ky} (requires (kx+1)*(ky+1) points), "
                f"got {len(X)}"
            )

        # Record training start time
        start_time = time.time()

        # Extract x and y coordinates
        x_coords = X[:, 0]  # Velocity
        y_coords = X[:, 1]  # Draft

        # Build spline
        try:
            self._model = SmoothBivariateSpline(
                x_coords,
                y_coords,
                y,
                s=self.smoothing,
                kx=self.kx,
                ky=self.ky
            )
        except Exception as e:
            raise RuntimeError(
                f"Spline fitting failed: {str(e)}. "
                f"Try reducing kx/ky or increasing smoothing parameter."
            ) from e

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
            >>> predictions = spline.predict(X_test)
            >>> print(predictions)
            [22.5, 30.2]
        """
        # Check if fitted
        self._check_is_fitted()

        # Validate input shape
        self._validate_input_shape(X)

        # Record prediction start time
        start_time = time.time()

        # Extract coordinates
        x_query = X[:, 0]  # Velocity
        y_query = X[:, 1]  # Draft

        # Make predictions using ev() method (point-by-point evaluation)
        predictions = self._model.ev(x_query, y_query)

        # Record prediction time
        self.predict_time = time.time() - start_time

        return predictions

    def evaluate_grid(self, v_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Evaluate spline on a regular grid (faster than point-by-point).

        This is an optimized method for evaluating on regular grids.
        Much faster than calling predict() for gridded queries.

        Args:
            v_grid: 1D array of velocity values
            t_grid: 1D array of draft values

        Returns:
            R_grid: 2D array of shape (len(v_grid), len(t_grid))

        Example:
            >>> v_grid = np.linspace(10, 25, 50)
            >>> t_grid = np.linspace(6, 10, 40)
            >>> R_grid = spline.evaluate_grid(v_grid, t_grid)
            >>> print(R_grid.shape)  # (50, 40)
        """
        self._check_is_fitted()

        # Use __call__ method which is optimized for grids
        # SmoothBivariateSpline returns shape (len(v_grid), len(t_grid)) directly
        R_grid = self._model(v_grid, t_grid, grid=True)

        return R_grid

    def get_metadata(self) -> dict:
        """Get metadata including spline-specific parameters.

        Returns:
            Dictionary with base metadata plus:
            - smoothing: Smoothing factor
            - kx, ky: Spline degrees

        Example:
            >>> meta = spline.get_metadata()
            >>> print(f"Degrees: kx={meta['kx']}, ky={meta['ky']}")
            >>> print(f"Training time: {meta['train_time']:.4f}s")
        """
        metadata = super().get_metadata()
        metadata.update({
            'smoothing': self.smoothing,
            'kx': self.kx,
            'ky': self.ky
        })
        return metadata

    def __repr__(self) -> str:
        """String representation with spline-specific info."""
        if self.is_fitted:
            return (
                f"Spline(kx={self.kx}, ky={self.ky}, smoothing={self.smoothing}, "
                f"fitted on {self.n_training_samples} samples, "
                f"train_time={self.train_time:.3f}s)"
            )
        else:
            return (
                f"Spline(kx={self.kx}, ky={self.ky}, smoothing={self.smoothing}, "
                f"not fitted)"
            )
