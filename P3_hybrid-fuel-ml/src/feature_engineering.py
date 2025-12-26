"""Feature engineering module for ship fuel efficiency prediction.

This module creates physics-based and interaction features to improve
model performance beyond raw features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Feature engineering for ship fuel consumption prediction.

    Creates physics-informed features based on domain knowledge:
    - Fuel rate (efficiency metric)
    - CO2 intensity (fuel quality proxy)
    - Efficiency deviation from fleet average
    - Interaction terms (distance × weather, etc.)

    Example:
        >>> fe = FeatureEngineer()
        >>> df_enriched = fe.create_all_features(df)
        >>> print(f"Original: {len(df.columns)} features")
        >>> print(f"Enriched: {len(df_enriched.columns)} features")
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.ship_type_efficiency_means = {}

    def create_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate physics-based features from domain knowledge.

        Args:
            df: Input DataFrame with raw features

        Returns:
            DataFrame with additional physics-based features

        Features created:
        - fuel_rate: fuel_consumption / distance (tonnes/nm)
        - co2_intensity: CO2_emissions / fuel_consumption (quality proxy)
        - distance_squared: distance^2 (non-linear effects)
        - efficiency_reciprocal: 1 / engine_efficiency (inverse relationship)
        """
        df = df.copy()

        # Fuel efficiency rate (tonnes per nautical mile)
        if 'fuel_consumption' in df.columns and 'distance' in df.columns:
            df['fuel_rate'] = df['fuel_consumption'] / df['distance']
            # Handle division by zero
            df['fuel_rate'] = df['fuel_rate'].replace([np.inf, -np.inf], np.nan)
            df['fuel_rate'] = df['fuel_rate'].fillna(df['fuel_rate'].median())

        # CO2 intensity (proxy for fuel quality/combustion efficiency)
        if 'CO2_emissions' in df.columns and 'fuel_consumption' in df.columns:
            df['co2_intensity'] = df['CO2_emissions'] / df['fuel_consumption']
            df['co2_intensity'] = df['co2_intensity'].replace([np.inf, -np.inf], np.nan)
            df['co2_intensity'] = df['co2_intensity'].fillna(df['co2_intensity'].median())

        # Distance squared (capture non-linear relationships)
        if 'distance' in df.columns:
            df['distance_squared'] = df['distance'] ** 2

        # Engine efficiency reciprocal (fuel ∝ 1/efficiency)
        if 'engine_efficiency' in df.columns:
            # Avoid division by zero
            df['efficiency_reciprocal'] = 1.0 / (df['engine_efficiency'] / 100.0)
            df['efficiency_reciprocal'] = df['efficiency_reciprocal'].replace([np.inf, -np.inf], np.nan)
            df['efficiency_reciprocal'] = df['efficiency_reciprocal'].fillna(df['efficiency_reciprocal'].median())

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction terms for non-linear patterns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features

        Interactions created:
        - distance_x_weather: Longer voyages amplify weather impact
        - distance_x_efficiency: Efficiency matters more on long voyages
        - weather_x_fuel_type: Different fuel types respond differently to weather
        """
        df = df.copy()

        # Distance × Weather (longer voyages more sensitive to weather)
        if 'distance' in df.columns and 'weather_ordinal' in df.columns:
            df['distance_x_weather'] = df['distance'] * df['weather_ordinal']

        # Distance × Engine Efficiency
        if 'distance' in df.columns and 'engine_efficiency' in df.columns:
            df['distance_x_efficiency'] = df['distance'] * (df['engine_efficiency'] / 100.0)

        # Weather × Fuel Type (HFO vs Diesel may respond differently to weather)
        if 'weather_ordinal' in df.columns and 'fuel_type_hfo' in df.columns:
            df['weather_x_fuel_type'] = df['weather_ordinal'] * df['fuel_type_hfo']

        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived statistical features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with derived features
        """
        df = df.copy()

        # Efficiency deviation from ship type average
        # (This requires fitting on train set first)
        if 'engine_efficiency' in df.columns:
            # Check if we have ship_type columns (one-hot encoded)
            ship_type_cols = [col for col in df.columns if col.startswith('ship_type_')]

            if ship_type_cols and len(self.ship_type_efficiency_means) > 0:
                # Determine which ship type this row belongs to
                df['efficiency_deviation'] = 0.0

                for ship_type_col in ship_type_cols:
                    ship_type = ship_type_col.replace('ship_type_', '')
                    mask = df[ship_type_col] == 1

                    if ship_type in self.ship_type_efficiency_means and mask.sum() > 0:
                        mean_eff = self.ship_type_efficiency_means[ship_type]
                        df.loc[mask, 'efficiency_deviation'] = (
                            df.loc[mask, 'engine_efficiency'] - mean_eff
                        )

        return df

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit feature engineer on training data.

        Computes statistics needed for derived features:
        - Ship type efficiency means

        Args:
            df: Training DataFrame

        Returns:
            Self for chaining
        """
        # Compute ship type efficiency means
        if 'engine_efficiency' in df.columns:
            ship_type_cols = [col for col in df.columns if col.startswith('ship_type_')]

            for ship_type_col in ship_type_cols:
                ship_type = ship_type_col.replace('ship_type_', '')
                mask = df[ship_type_col] == 1

                if mask.sum() > 0:
                    self.ship_type_efficiency_means[ship_type] = (
                        df.loc[mask, 'engine_efficiency'].mean()
                    )

        return self

    def create_all_features(self, df: pd.DataFrame,
                           include_interactions: bool = True,
                           include_derived: bool = True) -> pd.DataFrame:
        """Create all engineered features.

        Args:
            df: Input DataFrame
            include_interactions: Whether to create interaction terms
            include_derived: Whether to create derived features

        Returns:
            DataFrame with all engineered features

        Example:
            >>> fe = FeatureEngineer()
            >>> fe.fit(train_df)  # Fit on training data
            >>> train_enriched = fe.create_all_features(train_df)
            >>> val_enriched = fe.create_all_features(val_df)  # Use fitted stats
        """
        df = self.create_physics_features(df)

        if include_interactions:
            df = self.create_interaction_features(df)

        if include_derived:
            df = self.create_derived_features(df)

        return df

    def get_feature_names(self, df: pd.DataFrame,
                         exclude_target: bool = True,
                         exclude_leakage: bool = True) -> List[str]:
        """Get list of feature names for modeling.

        Args:
            df: DataFrame with features
            exclude_target: Exclude target variable (fuel_consumption)
            exclude_leakage: Exclude features that leak target info (CO2_emissions, fuel_rate)

        Returns:
            List of feature column names
        """
        exclude_cols = []

        if exclude_target:
            exclude_cols.extend(['fuel_consumption'])

        if exclude_leakage:
            # CO2 is directly derived from fuel consumption
            exclude_cols.extend(['CO2_emissions', 'co2_intensity'])
            # fuel_rate contains target in numerator
            if 'fuel_consumption' in df.columns:  # Only exclude if we have target
                exclude_cols.extend(['fuel_rate'])

        features = [col for col in df.columns if col not in exclude_cols]

        return features


def main():
    """Example usage of feature engineering."""
    print("=" * 80)
    print("FEATURE ENGINEERING EXAMPLE")
    print("=" * 80)

    # Load preprocessed data
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')

    print(f"\nOriginal features: {len(train.columns)}")

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Fit on training data
    fe.fit(train)

    # Create features for all splits
    train_enriched = fe.create_all_features(train)
    val_enriched = fe.create_all_features(val)
    test_enriched = fe.create_all_features(test)

    print(f"Enriched features: {len(train_enriched.columns)}")
    print(f"Features added: {len(train_enriched.columns) - len(train.columns)}")

    # Get feature names for modeling
    feature_names = fe.get_feature_names(train_enriched,
                                         exclude_target=True,
                                         exclude_leakage=True)

    print(f"\nFeatures for modeling: {len(feature_names)}")
    print("\nNew engineered features:")
    new_features = [col for col in train_enriched.columns if col not in train.columns]
    for i, feat in enumerate(new_features, 1):
        print(f"  {i}. {feat}")

    # Save enriched data
    train_enriched.to_csv('data/processed/train_enriched.csv', index=False)
    val_enriched.to_csv('data/processed/val_enriched.csv', index=False)
    test_enriched.to_csv('data/processed/test_enriched.csv', index=False)

    print(f"\n✓ Enriched datasets saved to data/processed/")
    print("=" * 80)


if __name__ == "__main__":
    main()
