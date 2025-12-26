"""Unit tests for uncertainty quantification models."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.models.uncertainty import QuantileRegressionUncertainty, BootstrapUncertainty
from src.calibration import CalibrationAnalyzer


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        'distance': np.random.uniform(50, 200, n_samples),
        'engine_efficiency': np.random.uniform(70, 95, n_samples),
        'weather_ordinal': np.random.choice([0, 1, 2], n_samples),
        'fuel_type_hfo': np.random.choice([0, 1], n_samples)
    })

    y = pd.Series(
        X['distance'] * 25 +
        (100 - X['engine_efficiency']) * 50 +
        np.random.normal(0, 200, n_samples)
    )

    return X, y


@pytest.fixture
def feature_names():
    """Feature names."""
    return ['distance', 'engine_efficiency', 'weather_ordinal', 'fuel_type_hfo']


class TestQuantileRegressionUncertainty:
    """Tests for quantile regression model."""

    def test_train_creates_all_quantile_models(self, sample_data, feature_names):
        """Verify all quantile models are trained."""
        X, y = sample_data

        model = QuantileRegressionUncertainty(quantiles=(0.05, 0.5, 0.95))
        model.train(X, y, feature_names=feature_names)

        assert model.is_fitted
        assert len(model.models) == 3
        assert 0.05 in model.models
        assert 0.5 in model.models
        assert 0.95 in model.models

    def test_predict_returns_correct_shapes(self, sample_data, feature_names):
        """Verify prediction output shapes."""
        X, y = sample_data

        model = QuantileRegressionUncertainty()
        model.train(X, y, feature_names=feature_names)

        mean, lower, upper = model.predict_with_intervals(X)

        assert mean.shape == (len(X),)
        assert lower.shape == (len(X),)
        assert upper.shape == (len(X),)

    def test_quantile_ordering(self, sample_data, feature_names):
        """Verify lower <= mean <= upper for all predictions."""
        X, y = sample_data

        model = QuantileRegressionUncertainty()
        model.train(X, y, feature_names=feature_names)

        mean, lower, upper = model.predict_with_intervals(X)

        # After monotonicity enforcement
        assert np.all(lower <= mean), "Lower bound should be <= mean"
        assert np.all(mean <= upper), "Mean should be <= upper bound"

    def test_save_and_load(self, sample_data, feature_names):
        """Verify model can be saved and loaded."""
        X, y = sample_data

        model = QuantileRegressionUncertainty()
        model.train(X, y, feature_names=feature_names)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)
            loaded_model = QuantileRegressionUncertainty.load(filepath)

            orig_mean, orig_lower, orig_upper = model.predict_with_intervals(X)
            load_mean, load_lower, load_upper = loaded_model.predict_with_intervals(X)

            np.testing.assert_array_almost_equal(orig_mean, load_mean)
            np.testing.assert_array_almost_equal(orig_lower, load_lower)

        finally:
            os.unlink(filepath)


class TestBootstrapUncertainty:
    """Tests for bootstrap ensemble model."""

    def test_train_creates_correct_number_of_models(self, sample_data, feature_names):
        """Verify correct number of bootstrap models."""
        X, y = sample_data

        n_bootstrap = 10
        model = BootstrapUncertainty(n_bootstrap=n_bootstrap)
        model.train(X, y, feature_names=feature_names)

        assert model.is_fitted
        assert len(model.models) == n_bootstrap

    def test_predict_returns_correct_shapes(self, sample_data, feature_names):
        """Verify prediction output shapes."""
        X, y = sample_data

        model = BootstrapUncertainty(n_bootstrap=5)
        model.train(X, y, feature_names=feature_names)

        mean, lower, upper = model.predict_with_intervals(X)

        assert mean.shape == (len(X),)
        assert lower.shape == (len(X),)
        assert upper.shape == (len(X),)

    def test_bootstrap_variance_nonzero(self, sample_data, feature_names):
        """Verify bootstrap predictions have variance."""
        X, y = sample_data

        model = BootstrapUncertainty(n_bootstrap=10)
        model.train(X, y, feature_names=feature_names)

        std = model.get_prediction_std(X)

        assert std.shape == (len(X),)
        assert np.mean(std) > 0, "Bootstrap should have non-zero variance"

    def test_interval_ordering(self, sample_data, feature_names):
        """Verify lower <= mean <= upper."""
        X, y = sample_data

        model = BootstrapUncertainty(n_bootstrap=10)
        model.train(X, y, feature_names=feature_names)

        mean, lower, upper = model.predict_with_intervals(X)

        assert np.all(lower <= mean), "Lower should be <= mean"
        assert np.all(mean <= upper), "Mean should be <= upper"


class TestCalibrationAnalyzer:
    """Tests for calibration analysis."""

    def test_compute_coverage(self):
        """Test coverage computation."""
        y_true = np.array([100, 200, 300, 400])
        lower = np.array([90, 180, 350, 380])  # 3rd outside
        upper = np.array([110, 220, 280, 420])  # 3rd outside

        coverage = CalibrationAnalyzer.compute_coverage(y_true, lower, upper)

        assert coverage == 0.75  # 3/4 within

    def test_compute_sharpness(self):
        """Test sharpness (mean width) computation."""
        lower = np.array([100, 200, 300])
        upper = np.array([200, 300, 400])

        mpiw = CalibrationAnalyzer.compute_sharpness(lower, upper)

        assert mpiw == 100  # Mean width is 100

    def test_perfect_coverage(self):
        """Test perfect calibration scenario."""
        y_true = np.array([100, 200, 300])
        lower = np.array([50, 150, 250])
        upper = np.array([150, 250, 350])

        coverage = CalibrationAnalyzer.compute_coverage(y_true, lower, upper)

        assert coverage == 1.0

    def test_zero_coverage(self):
        """Test worst case scenario."""
        y_true = np.array([100, 200, 300])
        lower = np.array([200, 300, 400])  # All lower > y_true
        upper = np.array([250, 350, 450])

        coverage = CalibrationAnalyzer.compute_coverage(y_true, lower, upper)

        assert coverage == 0.0

    def test_compute_all_metrics(self):
        """Test full metrics computation."""
        y_true = np.array([100, 200, 300, 400])
        predictions = np.array([110, 190, 310, 390])
        lower = np.array([50, 140, 260, 340])
        upper = np.array([150, 240, 360, 440])

        metrics = CalibrationAnalyzer.compute_all_metrics(
            y_true, predictions, lower, upper, confidence=0.90
        )

        assert 'PICP' in metrics
        assert 'MPIW' in metrics
        assert 'R2' in metrics
        assert 'RMSE' in metrics
        assert metrics['PICP'] == 1.0  # All within


def test_uncertainty_provides_intervals(sample_data, feature_names):
    """Integration test: uncertainty models provide meaningful intervals."""
    X, y = sample_data

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    model = QuantileRegressionUncertainty()
    model.train(X_train, y_train, feature_names=feature_names)

    # Predict with intervals
    mean, lower, upper = model.predict_with_intervals(X_test)

    # Intervals should have positive width
    widths = upper - lower
    assert np.all(widths > 0), "All intervals should have positive width"

    # At least some points should be within intervals
    coverage = CalibrationAnalyzer.compute_coverage(y_test.values, lower, upper)
    assert coverage > 0.2, "At least 20% of points should be in intervals"
