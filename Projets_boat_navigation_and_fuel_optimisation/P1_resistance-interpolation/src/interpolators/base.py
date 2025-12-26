"""Base interpolator class defining unified API for all interpolation methods.

This module provides the abstract base class that all interpolators must inherit from,
ensuring consistent interface across RBF, Splines, and Kriging methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import time


class BaseInterpolator(ABC):
    """Abstract base class for all interpolation methods.

    Enforces consistent API for training, prediction, and metadata retrieval.
    All interpolators must implement fit() and predict() methods.

    Attributes:
        name: Human-readable name of the interpolator
        is_fitted: Whether the interpolator has been trained
        train_time: Time taken to train (seconds)
        predict_time: Time taken for last prediction (seconds)
        n_training_samples: Number of training samples used

    Example:
        >>> # Concrete implementation (e.g., RBF)
        >>> interpolator = RBFInterpolator()
        >>> interpolator.fit(X_train, y_train)
        >>> predictions = interpolator.predict(X_test)
    """

    def __init__(self, name: str):
        """Initialize the base interpolator.

        Args:
            name: Human-readable name for this interpolator
        """
        self.name = name
        self.is_fitted = False
        self.train_time = None
        self.predict_time = None
        self.n_training_samples = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseInterpolator':
        """Train the interpolator on (X, y) pairs.

        Args:
            X: Training features (n_samples, n_features)
               For 2D interpolation: (n_samples, 2) with columns [V, T]
            y: Training targets (n_samples,)
               Resistance values

        Returns:
            self: Returns the instance for method chaining

        Raises:
            ValueError: If X or y have invalid shapes or contain invalid values

        Example:
            >>> interpolator.fit(X_train, y_train)
            >>> print(f"Trained on {interpolator.n_training_samples} samples")
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict resistance at query points.

        Args:
            X: Query features (n_queries, n_features)
               For 2D: (n_queries, 2) with columns [V, T]

        Returns:
            predictions: (n_queries,) array of predicted resistance values

        Raises:
            ValueError: If called before fit() or X has invalid shape

        Example:
            >>> predictions = interpolator.predict(X_test)
            >>> print(f"Made {len(predictions)} predictions")
        """
        pass

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """Convenience method for train + predict in one call.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features

        Returns:
            predictions: Predicted values on X_test

        Example:
            >>> predictions = interpolator.fit_predict(X_train, y_train, X_test)
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def get_metadata(self) -> Dict[str, any]:
        """Return metadata about the interpolator and its training.

        Returns:
            Dictionary containing:
            - name: Interpolator name
            - is_fitted: Whether training is complete
            - n_training_samples: Number of training samples (if fitted)
            - train_time: Training time in seconds (if fitted)
            - predict_time: Last prediction time in seconds (if predicted)

        Example:
            >>> meta = interpolator.get_metadata()
            >>> print(f"{meta['name']}: {meta['train_time']:.3f}s to train")
        """
        metadata = {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'n_training_samples': self.n_training_samples,
            'train_time': self.train_time,
            'predict_time': self.predict_time
        }
        return metadata

    def _validate_input_shape(self, X: np.ndarray, expected_features: int = 2) -> None:
        """Validate that input array has correct shape.

        Args:
            X: Input array to validate
            expected_features: Expected number of features (default: 2 for V, T)

        Raises:
            ValueError: If X doesn't have shape (n_samples, expected_features)
        """
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D array, got {X.ndim}D array with shape {X.shape}"
            )

        if X.shape[1] != expected_features:
            raise ValueError(
                f"X must have {expected_features} features, got {X.shape[1]} features"
            )

        if X.shape[0] == 0:
            raise ValueError("X cannot be empty (0 samples)")

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate training data before fitting.

        Args:
            X: Training features
            y: Training targets

        Raises:
            ValueError: If data is invalid
        """
        # Validate X shape
        self._validate_input_shape(X)

        # Validate y
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D array")

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length: X has {len(X)}, y has {len(y)}"
            )

        # Check for NaN or inf
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        if np.isnan(y).any():
            raise ValueError("y contains NaN values")
        if np.isinf(X).any():
            raise ValueError("X contains infinite values")
        if np.isinf(y).any():
            raise ValueError("y contains infinite values")

    def _check_is_fitted(self) -> None:
        """Check if the interpolator has been fitted.

        Raises:
            ValueError: If predict() is called before fit()
        """
        if not self.is_fitted:
            raise ValueError(
                f"{self.name} must be fitted before calling predict(). "
                f"Call fit(X, y) first."
            )

    def __repr__(self) -> str:
        """String representation of the interpolator.

        Returns:
            String describing the interpolator and its state
        """
        status = "fitted" if self.is_fitted else "not fitted"
        if self.is_fitted:
            return (
                f"{self.name}({status}, "
                f"n_samples={self.n_training_samples}, "
                f"train_time={self.train_time:.3f}s)"
            )
        else:
            return f"{self.name}({status})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
