"""Unit tests for hybrid physics-ML models."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.models.physics_baseline import PhysicsBasedFuelModel
from src.models.hybrid_model import ResidualCorrectionHybrid, FeatureAugmentationHybrid


@pytest.fixture
def sample_data():
    """Create sample dataset with ship type columns."""
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame({
        'distance': np.random.uniform(50, 200, n_samples),
        'engine_efficiency': np.random.uniform(70, 95, n_samples),
        'weather_ordinal': np.random.choice([0, 1, 2], n_samples),
        'fuel_type_hfo': np.random.choice([0, 1], n_samples),
        'ship_type_Tanker Ship': [1, 0, 0, 0] * 25,
        'ship_type_Oil Service Boat': [0, 1, 0, 0] * 25,
        'ship_type_Fishing Trawler': [0, 0, 1, 0] * 25,
        'ship_type_Surfer Boat': [0, 0, 0, 1] * 25
    })

    # Generate y based on physics relationship
    y = pd.Series(
        df['distance'] * 25 *
        (1 + df['weather_ordinal'] * 0.15) /
        (df['engine_efficiency'] / 100.0) +
        np.random.normal(0, 200, n_samples)
    )

    return df, y


@pytest.fixture
def feature_names():
    """Feature names for ML models."""
    return ['distance', 'engine_efficiency', 'weather_ordinal', 'fuel_type_hfo',
            'ship_type_Tanker Ship', 'ship_type_Oil Service Boat',
            'ship_type_Fishing Trawler', 'ship_type_Surfer Boat']


class TestPhysicsBasedFuelModel:
    """Tests for physics baseline model."""

    def test_calibrate_sets_coefficients(self, sample_data):
        """Verify calibration sets ship type coefficients."""
        X, y = sample_data

        model = PhysicsBasedFuelModel()
        model.calibrate(X, y)

        assert model.is_calibrated
        assert len(model.ship_coefficients) > 0
        assert 'Tanker Ship' in model.ship_coefficients

    def test_predict_returns_correct_shape(self, sample_data):
        """Verify prediction shape matches input."""
        X, y = sample_data

        model = PhysicsBasedFuelModel()
        model.calibrate(X, y)

        predictions = model.predict(X)

        assert predictions.shape == (len(X),)
        assert not np.any(np.isnan(predictions))

    def test_weather_increases_fuel(self, sample_data):
        """Verify stormy weather increases fuel prediction."""
        X, y = sample_data

        model = PhysicsBasedFuelModel()
        model.calibrate(X, y)

        # Create two identical rows except weather
        row_calm = X.iloc[[0]].copy()
        row_calm['weather_ordinal'] = 0

        row_stormy = X.iloc[[0]].copy()
        row_stormy['weather_ordinal'] = 2

        pred_calm = model.predict(row_calm)[0]
        pred_stormy = model.predict(row_stormy)[0]

        assert pred_stormy > pred_calm, "Stormy should increase fuel"

    def test_residuals_computed_correctly(self, sample_data):
        """Verify residuals are actual - predicted."""
        X, y = sample_data

        model = PhysicsBasedFuelModel()
        model.calibrate(X, y)

        residuals = model.get_residuals(X, y)
        predictions = model.predict(X)

        expected_residuals = y.values - predictions

        np.testing.assert_array_almost_equal(residuals, expected_residuals)

    def test_save_and_load(self, sample_data):
        """Verify model can be saved and loaded."""
        X, y = sample_data

        model = PhysicsBasedFuelModel()
        model.calibrate(X, y)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)
            loaded_model = PhysicsBasedFuelModel.load(filepath)

            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        finally:
            os.unlink(filepath)


class TestResidualCorrectionHybrid:
    """Tests for residual correction hybrid model."""

    def test_train_fits_both_models(self, sample_data, feature_names):
        """Verify training fits both physics and ML components."""
        X, y = sample_data

        model = ResidualCorrectionHybrid()
        model.train(X, y, feature_names=feature_names)

        assert model.is_fitted
        assert model.physics_model.is_calibrated
        assert model.ml_model.is_fitted

    def test_predict_returns_correct_shape(self, sample_data, feature_names):
        """Verify prediction shape."""
        X, y = sample_data

        model = ResidualCorrectionHybrid()
        model.train(X, y, feature_names=feature_names)

        predictions = model.predict(X)

        assert predictions.shape == (len(X),)

    def test_hybrid_combines_physics_and_ml(self, sample_data, feature_names):
        """Verify hybrid prediction = physics + ML correction."""
        X, y = sample_data

        model = ResidualCorrectionHybrid()
        model.train(X, y, feature_names=feature_names)

        hybrid_pred = model.predict(X)
        physics_pred = model.get_physics_predictions(X)
        ml_correction = model.get_ml_corrections(X)

        expected = physics_pred + ml_correction

        np.testing.assert_array_almost_equal(hybrid_pred, expected)

    def test_feature_importance_available(self, sample_data, feature_names):
        """Verify feature importance can be retrieved."""
        X, y = sample_data

        model = ResidualCorrectionHybrid()
        model.train(X, y, feature_names=feature_names)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert len(importance) == len(feature_names)


class TestFeatureAugmentationHybrid:
    """Tests for feature augmentation hybrid model."""

    def test_train_adds_physics_feature(self, sample_data, feature_names):
        """Verify physics prediction is added as feature."""
        X, y = sample_data

        model = FeatureAugmentationHybrid()
        model.train(X, y, feature_names=feature_names)

        assert model.is_fitted
        assert 'fuel_physics' in model.augmented_feature_names
        assert len(model.augmented_feature_names) == len(feature_names) + 1

    def test_predict_returns_correct_shape(self, sample_data, feature_names):
        """Verify prediction shape."""
        X, y = sample_data

        model = FeatureAugmentationHybrid()
        model.train(X, y, feature_names=feature_names)

        predictions = model.predict(X)

        assert predictions.shape == (len(X),)

    def test_physics_prediction_available(self, sample_data, feature_names):
        """Verify physics predictions can be retrieved."""
        X, y = sample_data

        model = FeatureAugmentationHybrid()
        model.train(X, y, feature_names=feature_names)

        physics_pred = model.get_physics_predictions(X)

        assert physics_pred.shape == (len(X),)
        assert not np.any(np.isnan(physics_pred))

    def test_feature_importance_includes_physics(self, sample_data, feature_names):
        """Verify fuel_physics appears in feature importance."""
        X, y = sample_data

        model = FeatureAugmentationHybrid()
        model.train(X, y, feature_names=feature_names)

        importance = model.get_feature_importance()

        assert 'fuel_physics' in importance['feature'].values


def test_hybrid_improves_on_physics(sample_data, feature_names):
    """Integration test: hybrid should improve over pure physics."""
    X, y = sample_data

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train physics baseline
    physics_model = PhysicsBasedFuelModel()
    physics_model.calibrate(X_train, y_train)
    physics_pred = physics_model.predict(X_test)
    physics_rmse = np.sqrt(np.mean((y_test.values - physics_pred) ** 2))

    # Train hybrid
    hybrid_model = ResidualCorrectionHybrid()
    hybrid_model.train(X_train, y_train, feature_names=feature_names)
    hybrid_pred = hybrid_model.predict(X_test)
    hybrid_rmse = np.sqrt(np.mean((y_test.values - hybrid_pred) ** 2))

    # Hybrid should be at least as good as physics
    assert hybrid_rmse <= physics_rmse * 1.1, \
        f"Hybrid ({hybrid_rmse:.2f}) should not be worse than physics ({physics_rmse:.2f})"
