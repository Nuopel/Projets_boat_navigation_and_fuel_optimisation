"""Linear regression models for fuel consumption prediction."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import joblib


class LinearFuelPredictor:
    """Ridge regression model for fuel consumption prediction.

    Uses L2 regularization to prevent overfitting while maintaining
    interpretability through linear coefficients.

    Example:
        >>> model = LinearFuelPredictor(alpha=1.0)
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> importance = model.get_feature_importance()
    """

    def __init__(self, alpha: float = 1.0, model_type: str = 'ridge'):
        """Initialize linear model.

        Args:
            alpha: Regularization strength (higher = more regularization)
            model_type: 'ridge' or 'lasso'
        """
        self.alpha = alpha
        self.model_type = model_type

        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray,
             feature_names: Optional[list] = None) -> 'LinearFuelPredictor':
        """Train model with feature scaling.

        Args:
            X: Training features (n_samples, n_features)
            y: Training target (n_samples,)
            feature_names: Optional list of feature names

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

        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted fuel consumption (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from coefficients.

        For linear models, absolute coefficient values indicate importance.

        Returns:
            DataFrame with features and their importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)

        return importance_df

    def get_params(self) -> dict:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            'alpha': self.alpha,
            'model_type': self.model_type,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'intercept': self.model.intercept_ if self.is_fitted else None
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
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'alpha': self.alpha,
            'model_type': self.model_type
        }

        joblib.dump(model_dict, filepath)
        print(f"✓ Model saved: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LinearFuelPredictor':
        """Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded LinearFuelPredictor instance
        """
        model_dict = joblib.load(filepath)

        instance = cls(alpha=model_dict['alpha'],
                      model_type=model_dict['model_type'])
        instance.model = model_dict['model']
        instance.scaler = model_dict['scaler']
        instance.feature_names = model_dict['feature_names']
        instance.is_fitted = True

        print(f"✓ Model loaded: {filepath}")
        return instance


def main():
    """Example usage of LinearFuelPredictor."""
    print("Linear model module loaded")
    print("Use LinearFuelPredictor class for Ridge/Lasso regression")


if __name__ == "__main__":
    main()
