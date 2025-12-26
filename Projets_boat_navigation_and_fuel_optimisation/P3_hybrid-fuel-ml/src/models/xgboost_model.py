"""XGBoost model for fuel consumption prediction."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
import joblib


class XGBoostFuelPredictor:
    """XGBoost gradient boosting model for fuel consumption prediction.

    Handles non-linear relationships and interactions automatically.
    Supports hyperparameter tuning via RandomizedSearchCV.

    Example:
        >>> model = XGBoostFuelPredictor()
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> importance = model.get_feature_importance()
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize XGBoost model.

        Args:
            params: XGBoost hyperparameters (None for defaults)
        """
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': 42,
            'n_jobs': -1
        }

        self.params = {**default_params, **(params or {})}
        self.model = XGBRegressor(**self.params)
        self.feature_names = None
        self.is_fitted = False
        self.best_params = None

    def train(self, X: np.ndarray, y: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             feature_names: Optional[list] = None,
             early_stopping_rounds: int = 10) -> 'XGBoostFuelPredictor':
        """Train model with optional early stopping.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target
            feature_names: Optional list of feature names
            early_stopping_rounds: Stop if no improvement for N rounds

        Returns:
            Self for method chaining
        """
        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Train with or without early stopping
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted fuel consumption
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                            n_iter: int = 20, cv: int = 5,
                            verbose: int = 1) -> Dict[str, Any]:
        """Tune hyperparameters using RandomizedSearchCV.

        Args:
            X: Training features
            y: Training target
            n_iter: Number of random combinations to try
            cv: Number of cross-validation folds
            verbose: Verbosity level

        Returns:
            Best hyperparameters found
        """
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Parameter distributions
        param_distributions = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 150, 200],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3]
        }

        # Base model for search
        base_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        # Randomized search
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            verbose=verbose,
            n_jobs=-1,
            random_state=42
        )

        print(f"Starting hyperparameter tuning ({n_iter} iterations, {cv}-fold CV)...")
        search.fit(X, y)

        # Store best parameters and update model
        self.best_params = search.best_params_
        self.params.update(self.best_params)
        self.model = XGBRegressor(**self.params)
        self.feature_names = feature_names

        print(f"\n✓ Best parameters found:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest CV RMSE: {-search.best_score_:.2f}")

        return self.best_params

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            DataFrame with features and their importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        # Get importance from model
        if importance_type == 'gain':
            importances = self.model.feature_importances_
        else:
            # For other types, use booster
            booster = self.model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            importances = np.array([
                importance_dict.get(f'f{i}', 0)
                for i in range(len(self.feature_names))
            ])

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df

    def get_params(self) -> dict:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            **self.params,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'best_params': self.best_params
        }

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save model (.pkl)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params,
            'best_params': self.best_params
        }

        joblib.dump(model_dict, filepath)
        print(f"✓ Model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'XGBoostFuelPredictor':
        """Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded XGBoostFuelPredictor instance
        """
        model_dict = joblib.load(filepath)

        instance = cls(params=model_dict['params'])
        instance.model = model_dict['model']
        instance.feature_names = model_dict['feature_names']
        instance.best_params = model_dict['best_params']
        instance.is_fitted = True

        print(f"✓ Model loaded: {filepath}")
        return instance


def main():
    """Example usage of XGBoostFuelPredictor."""
    print("XGBoost model module loaded")
    print("Use XGBoostFuelPredictor class for gradient boosting regression")


if __name__ == "__main__":
    main()
