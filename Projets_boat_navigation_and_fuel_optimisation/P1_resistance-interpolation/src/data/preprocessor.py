"""Data preprocessing module for yacht hydrodynamics dataset.

This module provides the DataPreprocessor class for deriving velocity and draft
proxies from the UCI Yacht Hydrodynamics dataset features.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocess yacht data to derive (Velocity, Draft) proxy variables.

    The UCI dataset contains dimensionless hydrodynamic parameters. For interpolation
    demonstration, we derive proxy physical variables:
    - Velocity (V): Derived from Froude number using V = Fr × √(g × L)
    - Draft (T): Derived from Beam-draught ratio using T = Beam / (B/T ratio)

    Note: These are proxies for demonstration. The interpolation methodology
    is identical for actual ship performance data.

    Attributes:
        reference_length: Reference length in meters (default: 10m for yacht)
        reference_beam: Reference beam width in meters (default: 3m)
        gravity: Gravitational acceleration in m/s² (default: 9.81)

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> df_vt = preprocessor.create_vt_surface(df_yacht)
        >>> print(df_vt.columns)
        ['V', 'T', 'R']
    """

    def __init__(
        self,
        reference_length: float = 10.0,
        reference_beam: float = 3.0,
        gravity: float = 9.81
    ):
        """Initialize the data preprocessor.

        Args:
            reference_length: Reference length for velocity calculation (meters)
            reference_beam: Reference beam width for draft calculation (meters)
            gravity: Gravitational acceleration (m/s²)
        """
        self.reference_length = reference_length
        self.reference_beam = reference_beam
        self.gravity = gravity

    def derive_velocity(self, froude: np.ndarray) -> np.ndarray:
        """Derive velocity from Froude number.

        The Froude number is defined as: Fr = V / √(g × L)
        Therefore: V = Fr × √(g × L)

        Result is converted to knots for maritime convention.

        Args:
            froude: Array of Froude numbers (dimensionless)

        Returns:
            Array of velocities in knots

        Example:
            >>> preprocessor = DataPreprocessor(reference_length=10.0)
            >>> v = preprocessor.derive_velocity(np.array([0.3, 0.4]))
            >>> print(v)  # velocities in knots
        """
        # V = Fr × √(g × L) gives velocity in m/s
        velocity_ms = froude * np.sqrt(self.gravity * self.reference_length)

        # Convert m/s to knots (1 m/s = 1.94384 knots)
        velocity_knots = velocity_ms * 1.94384

        return velocity_knots

    def derive_draft(self, beam_draft_ratio: np.ndarray) -> np.ndarray:
        """Derive draft from beam-to-draft ratio.

        Given: B/T ratio and assuming fixed beam width
        Therefore: T = Beam / (B/T ratio)

        Args:
            beam_draft_ratio: Array of beam-to-draft ratios (dimensionless)

        Returns:
            Array of draft values in meters

        Example:
            >>> preprocessor = DataPreprocessor(reference_beam=3.0)
            >>> t = preprocessor.derive_draft(np.array([3.0, 4.0]))
            >>> print(t)  # drafts in meters
        """
        # Avoid division by zero
        if np.any(beam_draft_ratio <= 0):
            raise ValueError("Beam-draft ratio must be positive")

        draft = self.reference_beam / beam_draft_ratio

        return draft

    def create_vt_surface(
        self,
        df: pd.DataFrame,
        aggregate_duplicates: bool = False,
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        """Create (V, T, R) surface from yacht hydrodynamics data.

        Extracts Froude number and Beam-draught ratio from the input DataFrame,
        derives velocity and draft, and combines with residuary resistance.

        Args:
            df: DataFrame with yacht data (must contain 'Froude_number',
                'Beam_draught_ratio', and 'Residuary_resistance' columns)
            aggregate_duplicates: If True, aggregate duplicate (V, T) pairs
                using agg_func.
            agg_func: Aggregation function name for duplicate (V, T) pairs.

        Returns:
            DataFrame with columns ['V', 'T', 'R'] representing:
            - V: Velocity in knots
            - T: Draft in meters
            - R: Residuary resistance (dimensionless)

        Raises:
            KeyError: If required columns are missing from input DataFrame
        """
        required_columns = ['Froude_number', 'Beam_draught_ratio', 'Residuary_resistance']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Derive velocity and draft
        velocity = self.derive_velocity(df['Froude_number'].values)
        draft = self.derive_draft(df['Beam_draught_ratio'].values)
        resistance = df['Residuary_resistance'].values

        # Create new DataFrame with V, T, R
        df_vt = pd.DataFrame({
            'V': velocity,
            'T': draft,
            'R': resistance
        })

        if aggregate_duplicates:
            df_vt = (
                df_vt
                .groupby(['V', 'T'], as_index=False)
                .agg({'R': agg_func})
            )

        return df_vt

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and test sets.

        Args:
            df: DataFrame with 'V', 'T', 'R' columns
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) where:
            - X_train, X_test: DataFrames with 'V', 'T' columns
            - y_train, y_test: Series with 'R' values

        Example:
            >>> preprocessor = DataPreprocessor()
            >>> df_vt = preprocessor.create_vt_surface(df_yacht)
            >>> X_train, X_test, y_train, y_test = preprocessor.train_test_split(df_vt)
            >>> print(X_train.shape)  # (246, 2) for 80/20 split of 308 samples
        """
        if 'R' not in df.columns:
            raise KeyError("DataFrame must contain 'R' (resistance) column")

        # Features: V and T
        X = df[['V', 'T']]
        # Target: R
        y = df['R']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def get_domain_bounds(self, df: pd.DataFrame) -> dict:
        """Get the bounds of the (V, T) domain.

        Args:
            df: DataFrame with 'V' and 'T' columns

        Returns:
            Dictionary with keys:
            - 'V_min', 'V_max': Velocity bounds
            - 'T_min', 'T_max': Draft bounds

        Example:
            >>> bounds = preprocessor.get_domain_bounds(df_vt)
            >>> print(f"Velocity range: {bounds['V_min']:.1f} - {bounds['V_max']:.1f} knots")
        """
        bounds = {
            'V_min': df['V'].min(),
            'V_max': df['V'].max(),
            'T_min': df['T'].min(),
            'T_max': df['T'].max()
        }

        return bounds

    def normalize_features(
        self,
        X: pd.DataFrame,
        fit_on: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, dict]:
        """Normalize features to [0, 1] range.

        Useful for interpolation methods sensitive to feature scaling.

        Args:
            X: DataFrame with features to normalize
            fit_on: Optional DataFrame to compute normalization parameters from.
                   If None, uses X itself (for training data).

        Returns:
            Tuple of:
            - Normalized DataFrame
            - Dictionary of normalization parameters (for inverse transform)

        Example:
            >>> X_train_norm, params = preprocessor.normalize_features(X_train)
            >>> X_test_norm, _ = preprocessor.normalize_features(X_test, fit_on=X_train)
        """
        if fit_on is None:
            fit_on = X

        params = {
            'V_min': fit_on['V'].min(),
            'V_max': fit_on['V'].max(),
            'T_min': fit_on['T'].min(),
            'T_max': fit_on['T'].max()
        }

        X_norm = pd.DataFrame({
            'V': (X['V'] - params['V_min']) / (params['V_max'] - params['V_min']),
            'T': (X['T'] - params['T_min']) / (params['T_max'] - params['T_min'])
        }, index=X.index)

        return X_norm, params

    def denormalize_features(self, X_norm: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Denormalize features from [0, 1] back to original range.

        Args:
            X_norm: Normalized DataFrame
            params: Normalization parameters from normalize_features()

        Returns:
            Denormalized DataFrame

        Example:
            >>> X_original = preprocessor.denormalize_features(X_norm, params)
        """
        X = pd.DataFrame({
            'V': X_norm['V'] * (params['V_max'] - params['V_min']) + params['V_min'],
            'T': X_norm['T'] * (params['T_max'] - params['T_min']) + params['T_min']
        }, index=X_norm.index)

        return X
