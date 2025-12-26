"""Tests for MetricsCalculator class."""

import pytest
import numpy as np
from src.evaluation.metrics import MetricsCalculator


def test_rmse_zero_error():
    """Test that RMSE is zero when predictions are perfect."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    rmse = MetricsCalculator.rmse(y_true, y_pred)

    assert rmse == pytest.approx(0.0, abs=1e-10), f"RMSE should be 0 for perfect predictions, got {rmse}"


def test_rmse_known_values():
    """Test RMSE with known values."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])  # Each prediction is off by 1

    # RMSE = sqrt(mean([1², 1², 1²])) = sqrt(1) = 1.0
    rmse = MetricsCalculator.rmse(y_true, y_pred)

    assert rmse == pytest.approx(1.0, abs=1e-10)


def test_mae_zero_error():
    """Test that MAE is zero when predictions are perfect."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    mae = MetricsCalculator.mae(y_true, y_pred)

    assert mae == pytest.approx(0.0, abs=1e-10)


def test_mae_known_values():
    """Test MAE with known values."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])  # Each prediction is off by 1

    # MAE = mean([|1|, |1|, |1|]) = 1.0
    mae = MetricsCalculator.mae(y_true, y_pred)

    assert mae == pytest.approx(1.0, abs=1e-10)


def test_mae_asymmetric_errors():
    """Test MAE handles positive and negative errors correctly."""
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0, -2.0])

    # MAE = mean([1, 1, 2, 2]) = 1.5
    mae = MetricsCalculator.mae(y_true, y_pred)

    assert mae == pytest.approx(1.5, abs=1e-10)


def test_r2_perfect_prediction():
    """Test that R² is 1.0 for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    r2 = MetricsCalculator.r2_score(y_true, y_pred)

    assert r2 == pytest.approx(1.0, abs=1e-10)


def test_r2_mean_baseline():
    """Test that R² is 0.0 when predictions equal mean of y_true."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.full_like(y_true, fill_value=np.mean(y_true))  # All predictions = mean

    r2 = MetricsCalculator.r2_score(y_true, y_pred)

    assert r2 == pytest.approx(0.0, abs=1e-6)


def test_r2_negative():
    """Test that R² can be negative for very bad predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # Very wrong predictions

    r2 = MetricsCalculator.r2_score(y_true, y_pred)

    assert r2 < 0, "R² should be negative for predictions worse than mean"


def test_max_error_identifies_worst():
    """Test that max_error identifies the worst-case error."""
    y_true = np.array([1.0, 2.0, 3.0, 10.0])
    y_pred = np.array([1.1, 2.0, 3.0, 5.0])  # Largest error is |10 - 5| = 5

    max_err = MetricsCalculator.max_error(y_true, y_pred)

    assert max_err == pytest.approx(5.0, abs=1e-10)


def test_max_error_symmetric():
    """Test that max_error considers absolute value."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, -4.0])  # Max absolute error is 4

    max_err = MetricsCalculator.max_error(y_true, y_pred)

    assert max_err == pytest.approx(4.0, abs=1e-10)


def test_mse_known_values():
    """Test MSE with known values."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])

    # MSE = mean([1², 1², 1²]) = 1.0
    mse = MetricsCalculator.mse(y_true, y_pred)

    assert mse == pytest.approx(1.0, abs=1e-10)


def test_mape_known_values():
    """Test MAPE with known values."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    # Errors: 10, 10, 10
    # Percentage errors: 10%, 5%, 3.33%
    # MAPE ≈ 6.11%
    mape = MetricsCalculator.mape(y_true, y_pred)

    expected_mape = np.mean([10.0, 5.0, 10.0/3.0])
    assert mape == pytest.approx(expected_mape, rel=0.01)


def test_mape_zero_handling():
    """Test that MAPE handles near-zero true values with epsilon."""
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.1, 1.1, 2.1])

    # Should not raise division by zero
    mape = MetricsCalculator.mape(y_true, y_pred, epsilon=1e-10)

    assert mape >= 0, "MAPE should be non-negative"
    assert not np.isnan(mape), "MAPE should not be NaN"
    assert not np.isinf(mape), "MAPE should not be infinite"


def test_metrics_nan_handling():
    """Test that metrics raise error on NaN inputs."""
    y_true = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1])

    with pytest.raises(ValueError, match="NaN"):
        MetricsCalculator.rmse(y_true, y_pred)

    with pytest.raises(ValueError, match="NaN"):
        MetricsCalculator.mae(y_true, y_pred)


def test_metrics_inf_handling():
    """Test that metrics raise error on infinite inputs."""
    y_true = np.array([1.0, 2.0, np.inf, 4.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1])

    with pytest.raises(ValueError, match="infinite"):
        MetricsCalculator.rmse(y_true, y_pred)


def test_metrics_shape_mismatch():
    """Test that metrics raise error on shape mismatch."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1])  # Different length

    with pytest.raises(ValueError, match="Shape mismatch"):
        MetricsCalculator.rmse(y_true, y_pred)


def test_metrics_empty_arrays():
    """Test that metrics raise error on empty arrays."""
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError, match="empty"):
        MetricsCalculator.rmse(y_true, y_pred)


def test_compute_all_metrics_keys():
    """Test that compute_all_metrics returns all expected keys."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1])

    metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred)

    expected_keys = {'rmse', 'mae', 'r2', 'max_error', 'mse', 'mape'}
    assert set(metrics.keys()) == expected_keys, \
        f"Missing keys: {expected_keys - set(metrics.keys())}"


def test_compute_all_metrics_consistency():
    """Test that compute_all_metrics matches individual metric functions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1])

    metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred)

    # Check each metric matches individual function
    assert metrics['rmse'] == pytest.approx(MetricsCalculator.rmse(y_true, y_pred))
    assert metrics['mae'] == pytest.approx(MetricsCalculator.mae(y_true, y_pred))
    assert metrics['r2'] == pytest.approx(MetricsCalculator.r2_score(y_true, y_pred))
    assert metrics['max_error'] == pytest.approx(MetricsCalculator.max_error(y_true, y_pred))
    assert metrics['mse'] == pytest.approx(MetricsCalculator.mse(y_true, y_pred))
    assert metrics['mape'] == pytest.approx(MetricsCalculator.mape(y_true, y_pred))


def test_rmse_vs_mae_large_outlier():
    """Test that RMSE penalizes large errors more than MAE."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 10.0])  # Last value has large error

    rmse = MetricsCalculator.rmse(y_true, y_pred)
    mae = MetricsCalculator.mae(y_true, y_pred)

    # RMSE should be larger than MAE due to squared errors
    assert rmse > mae, "RMSE should be larger than MAE when large errors exist"


def test_metrics_with_list_inputs():
    """Test that metrics work with list inputs (converted to numpy arrays)."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 2.1, 2.9, 4.1]

    # Should work without errors
    rmse = MetricsCalculator.rmse(y_true, y_pred)
    mae = MetricsCalculator.mae(y_true, y_pred)

    assert rmse > 0
    assert mae > 0


def test_metrics_with_integer_inputs():
    """Test that metrics work with integer inputs."""
    y_true = np.array([1, 2, 3, 4], dtype=int)
    y_pred = np.array([1, 2, 3, 5], dtype=int)

    # Errors: [0, 0, 0, 1]
    # MSE = mean([0, 0, 0, 1]) = 0.25
    # RMSE = sqrt(0.25) = 0.5
    # MAE = mean([0, 0, 0, 1]) = 0.25
    rmse = MetricsCalculator.rmse(y_true, y_pred)
    mae = MetricsCalculator.mae(y_true, y_pred)

    assert rmse == pytest.approx(0.5)
    assert mae == pytest.approx(0.25)


def test_validate_inputs_function():
    """Test the _validate_inputs helper function."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 2.1, 3.1]

    # Should convert to numpy and return
    y_true_arr, y_pred_arr = MetricsCalculator._validate_inputs(y_true, y_pred)

    assert isinstance(y_true_arr, np.ndarray)
    assert isinstance(y_pred_arr, np.ndarray)
    assert len(y_true_arr) == 3
    assert len(y_pred_arr) == 3
