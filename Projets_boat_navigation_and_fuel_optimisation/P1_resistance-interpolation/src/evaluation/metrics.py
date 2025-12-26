"""Metrics calculation module for interpolation evaluation.

This module provides the MetricsCalculator class for computing standardized
error metrics between predictions and ground truth.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MetricsCalculator:
    """Calculate standardized metrics for interpolation evaluation.

    Provides static methods for computing common regression metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)
    - Max Error (Worst-case prediction error)

    All methods handle edge cases and provide meaningful error messages.

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1])
        >>> rmse = MetricsCalculator.rmse(y_true, y_pred)
        >>> print(f"RMSE: {rmse:.3f}")
    """

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error.

        RMSE = √(mean((y_true - y_pred)²))

        Penalizes large errors more than MAE due to squaring.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            RMSE value (same units as y)

        Raises:
            ValueError: If arrays contain NaN or have mismatched shapes

        Example:
            >>> y_true = np.array([1, 2, 3])
            >>> y_pred = np.array([1.1, 2.0, 3.2])
            >>> MetricsCalculator.rmse(y_true, y_pred)
            0.1414...
        """
        y_true, y_pred = MetricsCalculator._validate_inputs(y_true, y_pred)
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error.

        MAE = mean(|y_true - y_pred|)

        More robust to outliers than RMSE.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAE value (same units as y)

        Raises:
            ValueError: If arrays contain NaN or have mismatched shapes

        Example:
            >>> y_true = np.array([1, 2, 3])
            >>> y_pred = np.array([2, 3, 4])
            >>> MetricsCalculator.mae(y_true, y_pred)
            1.0
        """
        y_true, y_pred = MetricsCalculator._validate_inputs(y_true, y_pred)
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² (Coefficient of Determination).

        R² = 1 - (SS_res / SS_tot)
        where SS_res = sum((y_true - y_pred)²)
              SS_tot = sum((y_true - mean(y_true))²)

        R² = 1.0: Perfect prediction
        R² = 0.0: Predictions are as good as mean of y_true
        R² < 0.0: Predictions are worse than mean

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            R² score (dimensionless, typically in range [-∞, 1])

        Raises:
            ValueError: If arrays contain NaN or have mismatched shapes

        Example:
            >>> y_true = np.array([1, 2, 3, 4])
            >>> y_pred = np.array([1, 2, 3, 4])  # Perfect prediction
            >>> MetricsCalculator.r2_score(y_true, y_pred)
            1.0
        """
        y_true, y_pred = MetricsCalculator._validate_inputs(y_true, y_pred)
        return r2_score(y_true, y_pred)

    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute maximum absolute error (worst-case error).

        max_error = max(|y_true - y_pred|)

        Useful for identifying worst-case prediction performance.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Maximum absolute error

        Raises:
            ValueError: If arrays contain NaN or have mismatched shapes

        Example:
            >>> y_true = np.array([1, 2, 3, 10])
            >>> y_pred = np.array([1, 2, 3, 5])
            >>> MetricsCalculator.max_error(y_true, y_pred)
            5.0
        """
        y_true, y_pred = MetricsCalculator._validate_inputs(y_true, y_pred)
        errors = np.abs(y_true - y_pred)
        return np.max(errors)

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error.

        MSE = mean((y_true - y_pred)²)

        Squared units, less interpretable than RMSE but faster to compute.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MSE value (squared units of y)

        Raises:
            ValueError: If arrays contain NaN or have mismatched shapes
        """
        y_true, y_pred = MetricsCalculator._validate_inputs(y_true, y_pred)
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Compute Mean Absolute Percentage Error.

        MAPE = mean(|y_true - y_pred| / (|y_true| + epsilon)) * 100

        Expressed as percentage. Adds epsilon to avoid division by zero.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            epsilon: Small constant to avoid division by zero

        Returns:
            MAPE value as percentage

        Raises:
            ValueError: If arrays contain NaN or have mismatched shapes

        Example:
            >>> y_true = np.array([100, 200, 300])
            >>> y_pred = np.array([110, 190, 310])
            >>> MetricsCalculator.mape(y_true, y_pred)
            5.0  # 5% average error
        """
        y_true, y_pred = MetricsCalculator._validate_inputs(y_true, y_pred)

        # Avoid division by zero
        denominator = np.abs(y_true) + epsilon
        percentage_errors = np.abs((y_true - y_pred) / denominator) * 100

        return np.mean(percentage_errors)

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all standard metrics in one call.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary with keys:
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'r2': R² score
            - 'max_error': Maximum absolute error
            - 'mse': Mean Squared Error
            - 'mape': Mean Absolute Percentage Error

        Raises:
            ValueError: If arrays contain NaN or have mismatched shapes

        Example:
            >>> y_true = np.array([1, 2, 3, 4])
            >>> y_pred = np.array([1.1, 2.1, 2.9, 4.1])
            >>> metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred)
            >>> for name, value in metrics.items():
            ...     print(f"{name}: {value:.4f}")
        """
        y_true, y_pred = MetricsCalculator._validate_inputs(y_true, y_pred)

        return {
            'rmse': MetricsCalculator.rmse(y_true, y_pred),
            'mae': MetricsCalculator.mae(y_true, y_pred),
            'r2': MetricsCalculator.r2_score(y_true, y_pred),
            'max_error': MetricsCalculator.max_error(y_true, y_pred),
            'mse': MetricsCalculator.mse(y_true, y_pred),
            'mape': MetricsCalculator.mape(y_true, y_pred)
        }

    @staticmethod
    def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Validate input arrays for metrics computation.

        Checks:
        - Both arrays are numpy arrays
        - No NaN values
        - Same shape
        - At least one element

        Args:
            y_true: Ground truth array
            y_pred: Prediction array

        Returns:
            Tuple of (y_true, y_pred) as numpy arrays

        Raises:
            ValueError: If validation fails
        """
        # Convert to numpy arrays if needed
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Check for NaN values
        if np.isnan(y_true).any():
            raise ValueError("y_true contains NaN values")
        if np.isnan(y_pred).any():
            raise ValueError("y_pred contains NaN values")

        # Check for inf values
        if np.isinf(y_true).any():
            raise ValueError("y_true contains infinite values")
        if np.isinf(y_pred).any():
            raise ValueError("y_pred contains infinite values")

        # Check shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, "
                f"y_pred has shape {y_pred.shape}"
            )

        # Check not empty
        if y_true.size == 0:
            raise ValueError("Arrays are empty")

        return y_true, y_pred
