"""Physics-based baseline model for fuel consumption prediction.

This model uses first-principles relationships from naval architecture
to predict fuel consumption based on domain knowledge.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.optimize import minimize
import joblib


class PhysicsBasedFuelModel:
    """Simplified physics model based on naval architecture principles.

    Uses the fundamental relationship:
    fuel = k × distance × weather_factor / efficiency_factor

    Where:
    - k is a ship-type specific coefficient (calibrated from data)
    - weather_factor accounts for increased resistance in bad weather
    - efficiency_factor normalizes engine efficiency

    This provides an interpretable baseline that captures known
    physical relationships before adding ML corrections.

    Example:
        >>> model = PhysicsBasedFuelModel()
        >>> model.calibrate(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self):
        """Initialize physics model."""
        # Ship-type specific coefficients (to be calibrated)
        self.ship_coefficients: Dict[str, float] = {}

        # Weather factors based on domain knowledge
        # Stormy conditions increase resistance/fuel by ~40-60%
        self.weather_factors = {
            0: 1.0,    # Calm
            1: 1.15,   # Moderate (+15%)
            2: 1.35    # Stormy (+35%)
        }

        # Fuel type factors (HFO slightly less efficient than Diesel)
        self.fuel_type_factors = {
            0: 1.0,    # Diesel
            1: 1.05    # HFO (+5%)
        }

        # Default coefficient for unknown ship types
        self.default_coefficient = 25.0

        self.is_calibrated = False

    def _get_ship_type(self, row: pd.Series) -> str:
        """Extract ship type from one-hot encoded columns.

        Args:
            row: Single row of data

        Returns:
            Ship type name
        """
        ship_type_cols = [col for col in row.index if col.startswith('ship_type_')]

        for col in ship_type_cols:
            if row[col] == 1:
                return col.replace('ship_type_', '')

        return 'Unknown'

    def _compute_physics_prediction(self, row: pd.Series) -> float:
        """Compute physics-based prediction for single observation.

        Args:
            row: Single row of data

        Returns:
            Predicted fuel consumption
        """
        # Extract features
        distance = row.get('distance', 100)
        engine_efficiency = row.get('engine_efficiency', 85)
        weather = int(row.get('weather_ordinal', 0))
        fuel_type = int(row.get('fuel_type_hfo', 0))

        # Get ship type coefficient
        ship_type = self._get_ship_type(row)
        k = self.ship_coefficients.get(ship_type, self.default_coefficient)

        # Weather and fuel type factors
        weather_factor = self.weather_factors.get(weather, 1.0)
        fuel_factor = self.fuel_type_factors.get(fuel_type, 1.0)

        # Physics formula
        # fuel = k × distance × weather_factor × fuel_factor / (efficiency/100)
        efficiency_factor = engine_efficiency / 100.0

        fuel_pred = (k * distance * weather_factor * fuel_factor) / efficiency_factor

        return fuel_pred

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> 'PhysicsBasedFuelModel':
        """Calibrate ship-type coefficients from training data.

        Uses least squares optimization to find optimal k for each ship type.

        Args:
            X: Training features
            y: Training target (fuel consumption)

        Returns:
            Self for method chaining
        """
        print("Calibrating physics model...")

        # Get unique ship types
        ship_type_cols = [col for col in X.columns if col.startswith('ship_type_')]

        for ship_type_col in ship_type_cols:
            ship_type = ship_type_col.replace('ship_type_', '')
            mask = X[ship_type_col] == 1

            if mask.sum() == 0:
                continue

            X_subset = X[mask]
            y_subset = y[mask]

            # Optimize coefficient k for this ship type
            def loss(k):
                predictions = []
                for _, row in X_subset.iterrows():
                    # Temporary set coefficient
                    self.ship_coefficients[ship_type] = k[0]
                    pred = self._compute_physics_prediction(row)
                    predictions.append(pred)
                return np.mean((y_subset.values - np.array(predictions)) ** 2)

            # Initial guess
            k0 = [self.default_coefficient]

            # Optimize
            result = minimize(loss, k0, method='L-BFGS-B',
                            bounds=[(1, 100)])

            self.ship_coefficients[ship_type] = result.x[0]
            print(f"  {ship_type}: k = {result.x[0]:.2f}")

        self.is_calibrated = True
        print("✓ Physics model calibrated")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using calibrated physics model.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted fuel consumption (n_samples,)
        """
        if not self.is_calibrated:
            raise ValueError("Model not calibrated. Call calibrate() first.")

        predictions = []
        for _, row in X.iterrows():
            pred = self._compute_physics_prediction(row)
            predictions.append(pred)

        return np.array(predictions)

    def get_residuals(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute residuals (actual - physics prediction).

        These residuals can be used to train an ML correction model.

        Args:
            X: Features
            y: Actual fuel consumption

        Returns:
            Residuals (n_samples,)
        """
        y_physics = self.predict(X)
        return y.values - y_physics

    def get_params(self) -> dict:
        """Get model parameters.

        Returns:
            Dictionary of calibrated parameters
        """
        return {
            'ship_coefficients': self.ship_coefficients,
            'weather_factors': self.weather_factors,
            'fuel_type_factors': self.fuel_type_factors,
            'default_coefficient': self.default_coefficient
        }

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save model (.pkl)
        """
        if not self.is_calibrated:
            raise ValueError("Cannot save uncalibrated model")

        model_dict = {
            'ship_coefficients': self.ship_coefficients,
            'weather_factors': self.weather_factors,
            'fuel_type_factors': self.fuel_type_factors,
            'default_coefficient': self.default_coefficient
        }

        joblib.dump(model_dict, filepath)
        print(f"✓ Physics model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'PhysicsBasedFuelModel':
        """Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded PhysicsBasedFuelModel instance
        """
        model_dict = joblib.load(filepath)

        instance = cls()
        instance.ship_coefficients = model_dict['ship_coefficients']
        instance.weather_factors = model_dict['weather_factors']
        instance.fuel_type_factors = model_dict['fuel_type_factors']
        instance.default_coefficient = model_dict['default_coefficient']
        instance.is_calibrated = True

        print(f"✓ Physics model loaded: {filepath}")
        return instance


def main():
    """Example usage of PhysicsBasedFuelModel."""
    print("Physics baseline model module loaded")
    print("Use PhysicsBasedFuelModel for interpretable fuel prediction")


if __name__ == "__main__":
    main()
