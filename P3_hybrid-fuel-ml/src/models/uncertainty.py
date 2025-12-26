"""Uncertainty quantification for fuel consumption predictions.

This module provides methods to estimate prediction uncertainty:
- Quantile Regression: Predict percentiles (5%, 50%, 95%)
- Bootstrapping: Ensemble variance from multiple models
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from xgboost import XGBRegressor
import joblib


class QuantileRegressionUncertainty:
    """Uncertainty estimation using quantile regression.

    Trains separate models for different quantiles to provide
    prediction intervals. For 90% CI, uses q05 and q95.

    Example:
        >>> model = QuantileRegressionUncertainty()
        >>> model.train(X_train, y_train)
        >>> mean, lower, upper = model.predict_with_intervals(X_test)
    """

    def __init__(self, quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95),
                 base_params: Optional[dict] = None):
        """Initialize quantile regression model.

        Args:
            quantiles: Quantiles to predict (default: 5%, 50%, 95%)
            base_params: XGBoost parameters for all quantile models
        """
        self.quantiles = quantiles

        default_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        self.base_params = {**default_params, **(base_params or {})}
        self.models = {}
        self.feature_names = None
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series,
             feature_names: Optional[list] = None) -> 'QuantileRegressionUncertainty':
        """Train quantile models.

        Args:
            X: Training features
            y: Training target
            feature_names: Feature names

        Returns:
            Self for method chaining
        """
        print("Training Quantile Regression models...")

        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = X

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y

        self.feature_names = feature_names

        for q in self.quantiles:
            print(f"  Training quantile {q:.2f}...")

            if q == 0.5:
                # Median - use standard squared error
                model = XGBRegressor(
                    objective='reg:squarederror',
                    **self.base_params
                )
            else:
                # Other quantiles - use quantile loss
                model = XGBRegressor(
                    objective='reg:quantileerror',
                    quantile_alpha=q,
                    **self.base_params
                )

            model.fit(X_arr, y_arr)
            self.models[q] = model

        self.is_fitted = True
        print("✓ Quantile Regression models trained")

        return self

    def predict_with_intervals(self, X: pd.DataFrame,
                               confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals.

        Args:
            X: Features
            confidence: Confidence level (default 0.90 for 90% CI)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        # Get quantile predictions
        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X_arr)

        # Median as point prediction
        if 0.5 in predictions:
            mean_pred = predictions[0.5]
        else:
            mean_pred = predictions[self.quantiles[len(self.quantiles) // 2]]

        # Confidence interval bounds
        alpha = 1 - confidence
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        # Find closest quantiles
        lower_pred = predictions[min(self.quantiles, key=lambda x: abs(x - lower_q))]
        upper_pred = predictions[min(self.quantiles, key=lambda x: abs(x - upper_q))]

        # Enforce monotonicity: lower <= mean <= upper
        lower_pred = np.minimum(lower_pred, mean_pred)
        upper_pred = np.maximum(upper_pred, mean_pred)

        return mean_pred, lower_pred, upper_pred

    def get_all_quantile_predictions(self, X: pd.DataFrame) -> dict:
        """Get predictions for all trained quantiles.

        Args:
            X: Features

        Returns:
            Dictionary of {quantile: predictions}
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        return {q: model.predict(X_arr) for q, model in self.models.items()}

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_dict = {
            'models': self.models,
            'quantiles': self.quantiles,
            'feature_names': self.feature_names,
            'base_params': self.base_params
        }

        joblib.dump(model_dict, filepath)
        print(f"✓ Quantile model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'QuantileRegressionUncertainty':
        """Load model from disk."""
        model_dict = joblib.load(filepath)

        instance = cls(quantiles=model_dict['quantiles'],
                      base_params=model_dict['base_params'])
        instance.models = model_dict['models']
        instance.feature_names = model_dict['feature_names']
        instance.is_fitted = True

        print(f"✓ Quantile model loaded: {filepath}")
        return instance


class BootstrapUncertainty:
    """Uncertainty estimation using bootstrap ensembles.

    Trains multiple models on bootstrap samples and uses
    prediction variance as uncertainty estimate.

    Example:
        >>> model = BootstrapUncertainty(n_bootstrap=30)
        >>> model.train(X_train, y_train)
        >>> mean, lower, upper = model.predict_with_intervals(X_test)
    """

    def __init__(self, n_bootstrap: int = 30,
                 base_params: Optional[dict] = None):
        """Initialize bootstrap model.

        Args:
            n_bootstrap: Number of bootstrap samples/models
            base_params: XGBoost parameters
        """
        self.n_bootstrap = n_bootstrap

        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        self.base_params = {**default_params, **(base_params or {})}
        self.models: List[XGBRegressor] = []
        self.feature_names = None
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series,
             feature_names: Optional[list] = None) -> 'BootstrapUncertainty':
        """Train bootstrap ensemble.

        Args:
            X: Training features
            y: Training target
            feature_names: Feature names

        Returns:
            Self for method chaining
        """
        print(f"Training Bootstrap ensemble ({self.n_bootstrap} models)...")

        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = X

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y

        self.feature_names = feature_names
        n_samples = len(X_arr)

        for i in range(self.n_bootstrap):
            if (i + 1) % 10 == 0:
                print(f"  Training model {i + 1}/{self.n_bootstrap}...")

            # Bootstrap sample (with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_arr[indices]
            y_boot = y_arr[indices]

            # Train model
            model = XGBRegressor(**self.base_params)
            model.set_params(random_state=42 + i)  # Different seed per model
            model.fit(X_boot, y_boot)

            self.models.append(model)

        self.is_fitted = True
        print("✓ Bootstrap ensemble trained")

        return self

    def predict_with_intervals(self, X: pd.DataFrame,
                               confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals.

        Args:
            X: Features
            confidence: Confidence level

        Returns:
            Tuple of (mean_predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        # Get predictions from all models
        all_predictions = np.array([
            model.predict(X_arr) for model in self.models
        ])

        # Mean prediction
        mean_pred = np.mean(all_predictions, axis=0)

        # Confidence interval from percentiles
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100

        lower_pred = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_pred = np.percentile(all_predictions, upper_percentile, axis=0)

        return mean_pred, lower_pred, upper_pred

    def get_prediction_std(self, X: pd.DataFrame) -> np.ndarray:
        """Get standard deviation of predictions across bootstrap models.

        Args:
            X: Features

        Returns:
            Standard deviation for each prediction
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        all_predictions = np.array([
            model.predict(X_arr) for model in self.models
        ])

        return np.std(all_predictions, axis=0)

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_dict = {
            'models': self.models,
            'n_bootstrap': self.n_bootstrap,
            'feature_names': self.feature_names,
            'base_params': self.base_params
        }

        joblib.dump(model_dict, filepath)
        print(f"✓ Bootstrap model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BootstrapUncertainty':
        """Load model from disk."""
        model_dict = joblib.load(filepath)

        instance = cls(n_bootstrap=model_dict['n_bootstrap'],
                      base_params=model_dict['base_params'])
        instance.models = model_dict['models']
        instance.feature_names = model_dict['feature_names']
        instance.is_fitted = True

        print(f"✓ Bootstrap model loaded: {filepath}")
        return instance


def main():
    """Example usage of uncertainty models."""
    print("Uncertainty quantification module loaded")
    print("Use QuantileRegressionUncertainty or BootstrapUncertainty")


if __name__ == "__main__":
    main()
