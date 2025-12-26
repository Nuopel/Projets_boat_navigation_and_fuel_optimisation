"""Kriging (Gaussian Process) interpolation for ship performance prediction.

This module provides Kriging interpolation using scikit-learn's GaussianProcessRegressor.
Kriging is unique in providing uncertainty quantification alongside predictions.
"""

import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from .base import BaseInterpolator


class KrigingInterpolator(BaseInterpolator):
    """Kriging (Gaussian Process Regression) for 2D resistance surfaces.

    Kriging treats the resistance surface as a realization of a Gaussian Process,
    providing both predictions and uncertainty estimates. It's the most statistically
    rigorous of the three interpolation methods.

    Advantages:
    - Uncertainty quantification (standard deviation at each point)
    - Probabilistic framework (confidence intervals)
    - Flexible kernel selection
    - Optimal predictions (BLUE: Best Linear Unbiased Estimator)
    - Handles noisy data gracefully

    Limitations:
    - Computationally expensive O(n³) for training
    - Slower than RBF/Splines for large datasets
    - Hyperparameter optimization can be time-consuming
    - Memory intensive for >1000 training points

    Attributes:
        kernel_type: Type of covariance kernel ('rbf', 'matern', 'rational_quadratic')
        length_scale: Initial length scale for kernel
        alpha: Noise level (regularization parameter)
        n_restarts_optimizer: Number of random starts for hyperparameter optimization
        normalize_y: Whether to normalize target values

    Example:
        >>> from src.interpolators.kriging import KrigingInterpolator
        >>> kriging = KrigingInterpolator(kernel_type='rbf', alpha=1e-6)
        >>> kriging.fit(X_train, y_train)
        >>> predictions, std_dev = kriging.predict(X_test, return_std=True)
        >>> print(f"Prediction: {predictions[0]:.2f} ± {2*std_dev[0]:.2f}")
    """

    def __init__(
        self,
        kernel_type: str = 'rbf',
        length_scale: float = 1.0,
        alpha: float = 1e-6,
        n_restarts_optimizer: int = 10,
        normalize_y: bool = True
    ):
        """Initialize Kriging interpolator.

        Args:
            kernel_type: Type of covariance kernel
                - 'rbf': Squared exponential (default, smooth)
                - 'matern': Matern kernel (more flexible)
                - 'rational_quadratic': Rational quadratic (mix of scales)
            length_scale: Initial length scale parameter
                - Larger values → smoother predictions
                - Smaller values → more local influence
                - Default: 1.0 (will be optimized)
            alpha: Noise level / regularization parameter
                - Larger values → more smoothing
                - Very small values → exact interpolation
                - Default: 1e-6 (nearly exact)
            n_restarts_optimizer: Number of random restarts for hyperparameter optimization
                - More restarts → better optimization, slower training
                - Default: 10 (good balance)
            normalize_y: Whether to normalize target values
                - Recommended: True for better numerical stability
                - Default: True

        Raises:
            ValueError: If kernel_type is not supported

        Example:
            >>> # Default RBF kernel with exact interpolation
            >>> kriging = KrigingInterpolator()
            >>>
            >>> # Matern kernel with more noise tolerance
            >>> kriging_noisy = KrigingInterpolator(
            ...     kernel_type='matern',
            ...     alpha=1e-4,
            ...     n_restarts_optimizer=20
            ... )
        """
        super().__init__(name='Kriging')
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self._model = None

        # Validate kernel type
        valid_kernels = ['rbf', 'matern', 'rational_quadratic']
        if kernel_type not in valid_kernels:
            raise ValueError(
                f"kernel_type must be one of {valid_kernels}, got '{kernel_type}'"
            )

    def _create_kernel(self):
        """Create the covariance kernel based on kernel_type.

        Returns:
            Sklearn kernel object with appropriate structure
        """
        # Constant kernel for overall variance scaling
        constant_kernel = C(1.0, (1e-3, 1e3))

        # Create base kernel
        if self.kernel_type == 'rbf':
            from sklearn.gaussian_process.kernels import RBF
            base_kernel = RBF(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-2, 1e2)
            )
        elif self.kernel_type == 'matern':
            from sklearn.gaussian_process.kernels import Matern
            base_kernel = Matern(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-2, 1e2),
                nu=1.5  # Smoothness parameter
            )
        elif self.kernel_type == 'rational_quadratic':
            from sklearn.gaussian_process.kernels import RationalQuadratic
            base_kernel = RationalQuadratic(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-2, 1e2),
                alpha=1.0
            )

        # Add white noise kernel for numerical stability
        noise_kernel = WhiteKernel(
            noise_level=self.alpha,
            noise_level_bounds=(1e-10, 1e-1)
        )

        # Combine kernels: C * base + noise
        kernel = constant_kernel * base_kernel + noise_kernel

        return kernel

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KrigingInterpolator':
        """Train Kriging interpolator on (V, T, R) data.

        This performs maximum likelihood estimation of hyperparameters
        followed by computation of the Kriging weights.

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
            >>> kriging = KrigingInterpolator()
            >>> kriging.fit(X_train, y_train)
            Kriging(fitted, n_samples=3, train_time=0.123s)
        """
        # Validate input data
        self._validate_training_data(X, y)

        # Record training start time
        start_time = time.time()

        # Create kernel
        kernel = self._create_kernel()

        # Build Gaussian Process model
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # Noise is handled by WhiteKernel
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=42  # For reproducible hyperparameter optimization
        )

        # Fit model
        self._model.fit(X, y)

        # Record training completion
        self.train_time = time.time() - start_time
        self.is_fitted = True
        self.n_training_samples = len(X)

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> np.ndarray:
        """Predict resistance at query points with optional uncertainty.

        Args:
            X: Query features (n_queries, 2) with columns [V, T]
            return_std: If True, return (predictions, std_dev) tuple
                       If False, return only predictions

        Returns:
            predictions: (n_queries,) predicted resistance values
            OR
            (predictions, std_dev): tuple if return_std=True
                - predictions: (n_queries,) predicted resistance
                - std_dev: (n_queries,) standard deviation (uncertainty)

        Raises:
            ValueError: If called before fit() or X has invalid shape

        Example:
            >>> X_test = np.array([[12, 6.5], [18, 7.5]])
            >>>
            >>> # Predictions only
            >>> predictions = kriging.predict(X_test)
            >>> print(predictions)
            [22.5, 30.2]
            >>>
            >>> # Predictions with uncertainty
            >>> predictions, std = kriging.predict(X_test, return_std=True)
            >>> for pred, uncertainty in zip(predictions, std):
            ...     print(f"{pred:.2f} ± {2*uncertainty:.2f}")
            22.50 ± 0.42
            30.20 ± 0.58
        """
        # Check if fitted
        self._check_is_fitted()

        # Validate input shape
        self._validate_input_shape(X)

        # Record prediction start time
        start_time = time.time()

        # Make predictions
        if return_std:
            predictions, std_dev = self._model.predict(X, return_std=True)
            result = (predictions, std_dev)
        else:
            predictions = self._model.predict(X, return_std=False)
            result = predictions

        # Record prediction time
        self.predict_time = time.time() - start_time

        return result

    def get_metadata(self) -> dict:
        """Get metadata including Kriging-specific parameters.

        Returns:
            Dictionary with base metadata plus:
            - kernel_type: Type of covariance kernel
            - alpha: Noise level
            - length_scale: Kernel length scale (optimized if fitted)
            - log_marginal_likelihood: Model evidence (if fitted)

        Example:
            >>> meta = kriging.get_metadata()
            >>> print(f"Kernel: {meta['kernel_type']}")
            >>> print(f"Log ML: {meta['log_marginal_likelihood']:.2f}")
        """
        metadata = super().get_metadata()
        metadata.update({
            'kernel_type': self.kernel_type,
            'alpha': self.alpha,
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'normalize_y': self.normalize_y
        })

        # Add fitted parameters if available
        if self.is_fitted:
            metadata['optimized_kernel'] = str(self._model.kernel_)
            metadata['log_marginal_likelihood'] = self._model.log_marginal_likelihood_value_

        return metadata

    def __repr__(self) -> str:
        """String representation with Kriging-specific info."""
        if self.is_fitted:
            return (
                f"Kriging(kernel={self.kernel_type}, alpha={self.alpha}, "
                f"fitted on {self.n_training_samples} samples, "
                f"train_time={self.train_time:.3f}s, "
                f"log_ML={self._model.log_marginal_likelihood_value_:.2f})"
            )
        else:
            return (
                f"Kriging(kernel={self.kernel_type}, alpha={self.alpha}, "
                f"not fitted)"
            )
