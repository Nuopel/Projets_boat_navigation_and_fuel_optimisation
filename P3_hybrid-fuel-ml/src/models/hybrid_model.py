"""Hybrid Physics-ML models for fuel consumption prediction.

This module implements two hybrid approaches that combine
physics-based modeling with machine learning corrections.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import joblib

from src.models.physics_baseline import PhysicsBasedFuelModel
from src.models.xgboost_model import XGBoostFuelPredictor


class ResidualCorrectionHybrid:
    """Hybrid model using residual correction approach.

    Architecture:
        fuel_pred = fuel_physics + ML_model(residuals)

    The physics model captures known relationships, while
    the ML model learns to correct systematic errors.

    Example:
        >>> hybrid = ResidualCorrectionHybrid()
        >>> hybrid.train(X_train, y_train, X_val, y_val, feature_names)
        >>> predictions = hybrid.predict(X_test)
    """

    def __init__(self, ml_params: Optional[Dict[str, Any]] = None):
        """Initialize hybrid model.

        Args:
            ml_params: XGBoost hyperparameters for correction model
        """
        self.physics_model = PhysicsBasedFuelModel()
        self.ml_model = XGBoostFuelPredictor(params=ml_params)
        self.feature_names = None
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             feature_names: Optional[list] = None) -> 'ResidualCorrectionHybrid':
        """Train hybrid model.

        Steps:
        1. Calibrate physics model
        2. Compute residuals (actual - physics)
        3. Train ML model to predict residuals

        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation target
            feature_names: Feature names for ML model

        Returns:
            Self for method chaining
        """
        print("\n" + "=" * 60)
        print("TRAINING: Residual Correction Hybrid")
        print("=" * 60)

        # Store feature names
        self.feature_names = feature_names or X.columns.tolist()

        # Step 1: Calibrate physics model
        self.physics_model.calibrate(X, y)

        # Step 2: Compute residuals
        y_physics = self.physics_model.predict(X)
        residuals = y.values - y_physics

        print(f"\nPhysics model RMSE: {np.sqrt(np.mean((y.values - y_physics)**2)):.2f}")
        print(f"Residual mean: {residuals.mean():.2f}")
        print(f"Residual std: {residuals.std():.2f}")

        # Step 3: Train ML model on residuals
        print("\nTraining ML correction model...")

        # Validation residuals if provided
        if X_val is not None and y_val is not None:
            y_physics_val = self.physics_model.predict(X_val)
            residuals_val = y_val.values - y_physics_val

            self.ml_model.train(
                X[self.feature_names], pd.Series(residuals),
                X_val[self.feature_names], pd.Series(residuals_val),
                feature_names=self.feature_names
            )
        else:
            self.ml_model.train(
                X[self.feature_names], pd.Series(residuals),
                feature_names=self.feature_names
            )

        self.is_fitted = True
        print("✓ Residual Correction Hybrid trained")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using hybrid model.

        Args:
            X: Features

        Returns:
            Predicted fuel consumption
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        # Physics prediction
        y_physics = self.physics_model.predict(X)

        # ML correction
        residual_pred = self.ml_model.predict(X[self.feature_names])

        # Combine
        return y_physics + residual_pred

    def get_physics_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get just the physics component predictions.

        Args:
            X: Features

        Returns:
            Physics model predictions
        """
        return self.physics_model.predict(X)

    def get_ml_corrections(self, X: pd.DataFrame) -> np.ndarray:
        """Get just the ML correction component.

        Args:
            X: Features

        Returns:
            ML correction predictions
        """
        return self.ml_model.predict(X[self.feature_names])

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from ML correction model.

        Returns:
            DataFrame with feature importance
        """
        return self.ml_model.get_feature_importance()

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_dict = {
            'physics_model': self.physics_model.get_params(),
            'ml_model': self.ml_model.model,
            'feature_names': self.feature_names,
            'ml_params': self.ml_model.params
        }

        joblib.dump(model_dict, filepath)
        print(f"✓ Hybrid model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ResidualCorrectionHybrid':
        """Load model from disk."""
        model_dict = joblib.load(filepath)

        instance = cls()

        # Restore physics model
        instance.physics_model.ship_coefficients = model_dict['physics_model']['ship_coefficients']
        instance.physics_model.weather_factors = model_dict['physics_model']['weather_factors']
        instance.physics_model.fuel_type_factors = model_dict['physics_model']['fuel_type_factors']
        instance.physics_model.is_calibrated = True

        # Restore ML model
        instance.ml_model.model = model_dict['ml_model']
        instance.ml_model.feature_names = model_dict['feature_names']
        instance.ml_model.params = model_dict['ml_params']
        instance.ml_model.is_fitted = True

        instance.feature_names = model_dict['feature_names']
        instance.is_fitted = True

        print(f"✓ Hybrid model loaded: {filepath}")
        return instance


class FeatureAugmentationHybrid:
    """Hybrid model using feature augmentation approach.

    Architecture:
        fuel_pred = ML_model([original_features, fuel_physics])

    The physics prediction is added as a feature, allowing
    the ML model to learn when to trust physics vs. data.

    Example:
        >>> hybrid = FeatureAugmentationHybrid()
        >>> hybrid.train(X_train, y_train, X_val, y_val, feature_names)
        >>> predictions = hybrid.predict(X_test)
    """

    def __init__(self, ml_params: Optional[Dict[str, Any]] = None):
        """Initialize hybrid model.

        Args:
            ml_params: XGBoost hyperparameters
        """
        self.physics_model = PhysicsBasedFuelModel()
        self.ml_model = XGBoostFuelPredictor(params=ml_params)
        self.feature_names = None
        self.augmented_feature_names = None
        self.is_fitted = False

    def _augment_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add physics prediction as feature.

        Args:
            X: Original features

        Returns:
            Augmented features with physics prediction
        """
        X_augmented = X.copy()
        X_augmented['fuel_physics'] = self.physics_model.predict(X)
        return X_augmented

    def train(self, X: pd.DataFrame, y: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             feature_names: Optional[list] = None) -> 'FeatureAugmentationHybrid':
        """Train hybrid model.

        Steps:
        1. Calibrate physics model
        2. Augment features with physics prediction
        3. Train ML model on augmented features

        Args:
            X: Training features
            y: Training target
            X_val: Validation features
            y_val: Validation target
            feature_names: Original feature names

        Returns:
            Self for method chaining
        """
        print("\n" + "=" * 60)
        print("TRAINING: Feature Augmentation Hybrid")
        print("=" * 60)

        # Store feature names
        self.feature_names = feature_names or X.columns.tolist()

        # Step 1: Calibrate physics model
        self.physics_model.calibrate(X, y)

        # Step 2: Augment features
        X_augmented = self._augment_features(X)

        # Augmented feature names
        self.augmented_feature_names = self.feature_names + ['fuel_physics']

        y_physics = X_augmented['fuel_physics'].values
        print(f"\nPhysics model RMSE: {np.sqrt(np.mean((y.values - y_physics)**2)):.2f}")

        # Step 3: Train ML model on augmented features
        print("\nTraining ML model with augmented features...")

        if X_val is not None and y_val is not None:
            X_val_augmented = self._augment_features(X_val)

            self.ml_model.train(
                X_augmented[self.augmented_feature_names], y,
                X_val_augmented[self.augmented_feature_names], y_val,
                feature_names=self.augmented_feature_names
            )
        else:
            self.ml_model.train(
                X_augmented[self.augmented_feature_names], y,
                feature_names=self.augmented_feature_names
            )

        self.is_fitted = True
        print("✓ Feature Augmentation Hybrid trained")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using hybrid model.

        Args:
            X: Features

        Returns:
            Predicted fuel consumption
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        # Augment features
        X_augmented = self._augment_features(X)

        # ML prediction on augmented features
        return self.ml_model.predict(X_augmented[self.augmented_feature_names])

    def get_physics_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get just the physics component predictions."""
        return self.physics_model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance including physics feature."""
        return self.ml_model.get_feature_importance()

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_dict = {
            'physics_model': self.physics_model.get_params(),
            'ml_model': self.ml_model.model,
            'feature_names': self.feature_names,
            'augmented_feature_names': self.augmented_feature_names,
            'ml_params': self.ml_model.params
        }

        joblib.dump(model_dict, filepath)
        print(f"✓ Hybrid model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeatureAugmentationHybrid':
        """Load model from disk."""
        model_dict = joblib.load(filepath)

        instance = cls()

        # Restore physics model
        instance.physics_model.ship_coefficients = model_dict['physics_model']['ship_coefficients']
        instance.physics_model.weather_factors = model_dict['physics_model']['weather_factors']
        instance.physics_model.fuel_type_factors = model_dict['physics_model']['fuel_type_factors']
        instance.physics_model.is_calibrated = True

        # Restore ML model
        instance.ml_model.model = model_dict['ml_model']
        instance.ml_model.feature_names = model_dict['augmented_feature_names']
        instance.ml_model.params = model_dict['ml_params']
        instance.ml_model.is_fitted = True

        instance.feature_names = model_dict['feature_names']
        instance.augmented_feature_names = model_dict['augmented_feature_names']
        instance.is_fitted = True

        print(f"✓ Hybrid model loaded: {filepath}")
        return instance


def main():
    """Example usage of hybrid models."""
    print("Hybrid model module loaded")
    print("Use ResidualCorrectionHybrid or FeatureAugmentationHybrid")


if __name__ == "__main__":
    main()
