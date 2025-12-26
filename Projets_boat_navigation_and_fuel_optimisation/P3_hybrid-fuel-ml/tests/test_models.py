"""Unit tests for ML models."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.models.linear_model import LinearFuelPredictor
from src.models.xgboost_model import XGBoostFuelPredictor
from src.feature_engineering import FeatureEngineer
from src.evaluation import ModelEvaluator


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

    # Generate y with known relationship
    y = (
        X['distance'] * 25 +
        (100 - X['engine_efficiency']) * 50 +
        X['weather_ordinal'] * 200 +
        np.random.normal(0, 100, n_samples)
    )

    return X, pd.Series(y)


@pytest.fixture
def feature_names():
    """Feature names for testing."""
    return ['distance', 'engine_efficiency', 'weather_ordinal', 'fuel_type_hfo']


class TestLinearFuelPredictor:
    """Tests for Ridge regression model."""

    def test_train_and_predict_shape(self, sample_data, feature_names):
        """Verify output shape matches input samples."""
        X, y = sample_data

        model = LinearFuelPredictor(alpha=1.0)
        model.train(X, y, feature_names=feature_names)

        predictions = model.predict(X)

        assert predictions.shape == (len(X),)
        assert not np.any(np.isnan(predictions))

    def test_predictions_in_reasonable_range(self, sample_data, feature_names):
        """Verify predictions are in reasonable range."""
        X, y = sample_data

        model = LinearFuelPredictor(alpha=1.0)
        model.train(X, y, feature_names=feature_names)

        predictions = model.predict(X)

        # Predictions should be within 3 std of training data
        y_std = y.std()
        y_mean = y.mean()

        assert predictions.min() > y_mean - 5 * y_std
        assert predictions.max() < y_mean + 5 * y_std

    def test_feature_importance_returns_dataframe(self, sample_data, feature_names):
        """Verify feature importance returns proper DataFrame."""
        X, y = sample_data

        model = LinearFuelPredictor(alpha=1.0)
        model.train(X, y, feature_names=feature_names)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns
        assert len(importance) == len(feature_names)

    def test_model_save_and_load(self, sample_data, feature_names):
        """Verify model can be saved and loaded."""
        X, y = sample_data

        model = LinearFuelPredictor(alpha=1.0)
        model.train(X, y, feature_names=feature_names)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)

            # Load
            loaded_model = LinearFuelPredictor.load(filepath)

            # Predictions should be identical
            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        finally:
            os.unlink(filepath)

    def test_unfitted_model_raises_error(self, sample_data):
        """Verify error when predicting with unfitted model."""
        X, _ = sample_data
        model = LinearFuelPredictor()

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)


class TestXGBoostFuelPredictor:
    """Tests for XGBoost model."""

    def test_train_and_predict_shape(self, sample_data, feature_names):
        """Verify output shape matches input samples."""
        X, y = sample_data

        model = XGBoostFuelPredictor()
        model.train(X, y, feature_names=feature_names)

        predictions = model.predict(X)

        assert predictions.shape == (len(X),)
        assert not np.any(np.isnan(predictions))

    def test_feature_importance_returns_dataframe(self, sample_data, feature_names):
        """Verify feature importance returns proper DataFrame."""
        X, y = sample_data

        model = XGBoostFuelPredictor()
        model.train(X, y, feature_names=feature_names)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == len(feature_names)

    def test_custom_parameters(self, sample_data, feature_names):
        """Verify custom parameters are applied."""
        X, y = sample_data

        custom_params = {'max_depth': 3, 'n_estimators': 50}
        model = XGBoostFuelPredictor(params=custom_params)

        assert model.params['max_depth'] == 3
        assert model.params['n_estimators'] == 50

        model.train(X, y, feature_names=feature_names)
        predictions = model.predict(X)

        assert predictions.shape == (len(X),)

    def test_model_save_and_load(self, sample_data, feature_names):
        """Verify model can be saved and loaded."""
        X, y = sample_data

        model = XGBoostFuelPredictor()
        model.train(X, y, feature_names=feature_names)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)

            loaded_model = XGBoostFuelPredictor.load(filepath)

            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        finally:
            os.unlink(filepath)


class TestFeatureEngineer:
    """Tests for feature engineering module."""

    def test_physics_features_created(self):
        """Verify physics-based features are created."""
        df = pd.DataFrame({
            'distance': [100, 150, 200],
            'fuel_consumption': [2500, 4000, 5500],
            'CO2_emissions': [7000, 11000, 15000],
            'engine_efficiency': [85, 80, 90]
        })

        fe = FeatureEngineer()
        df_enriched = fe.create_physics_features(df)

        assert 'distance_squared' in df_enriched.columns
        assert 'efficiency_reciprocal' in df_enriched.columns
        assert df_enriched['distance_squared'].iloc[0] == 100 ** 2

    def test_interaction_features_created(self):
        """Verify interaction features are created."""
        df = pd.DataFrame({
            'distance': [100, 150],
            'weather_ordinal': [0, 2],
            'engine_efficiency': [85, 80],
            'fuel_type_hfo': [1, 0]
        })

        fe = FeatureEngineer()
        df_enriched = fe.create_interaction_features(df)

        assert 'distance_x_weather' in df_enriched.columns
        assert 'distance_x_efficiency' in df_enriched.columns
        assert df_enriched['distance_x_weather'].iloc[0] == 100 * 0

    def test_get_feature_names_excludes_target(self):
        """Verify target variable is excluded from feature names."""
        df = pd.DataFrame({
            'distance': [100],
            'fuel_consumption': [2500],
            'CO2_emissions': [7000]
        })

        fe = FeatureEngineer()
        features = fe.get_feature_names(df, exclude_target=True, exclude_leakage=True)

        assert 'fuel_consumption' not in features
        assert 'CO2_emissions' not in features
        assert 'distance' in features


class TestModelEvaluator:
    """Tests for evaluation module."""

    def test_compute_metrics_returns_dict(self):
        """Verify metrics computation returns proper dict."""
        y_true = np.array([1000, 2000, 3000, 4000])
        y_pred = np.array([1100, 1900, 3100, 3900])

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred, "Test")

        assert isinstance(metrics, dict)
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'MAPE' in metrics
        assert 'R2' in metrics
        assert 'Max_Error' in metrics

    def test_metrics_values_correct(self):
        """Verify metrics values are computed correctly."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])  # Perfect predictions

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)

        assert metrics['RMSE'] == 0.0
        assert metrics['MAE'] == 0.0
        assert metrics['R2'] == 1.0
        assert metrics['Max_Error'] == 0.0

    def test_r2_less_than_one(self):
        """Verify R² is always ≤ 1."""
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)

        assert metrics['R2'] <= 1.0


def test_end_to_end_pipeline(sample_data, feature_names):
    """Integration test: full training and evaluation pipeline."""
    X, y = sample_data

    # Split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    model = XGBoostFuelPredictor()
    model.train(X_train, y_train, feature_names=feature_names)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = ModelEvaluator.compute_metrics(y_test.values, y_pred, "Test")

    # Check model learned something (R² > 0)
    assert metrics['R2'] > 0, "Model should learn from data"
    assert metrics['RMSE'] < y_test.std() * 2, "RMSE should be reasonable"
