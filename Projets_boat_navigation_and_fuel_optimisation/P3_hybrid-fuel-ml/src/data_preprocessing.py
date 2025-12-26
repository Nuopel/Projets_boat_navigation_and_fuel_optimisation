"""Data preprocessing pipeline for ship fuel efficiency dataset.

This module handles data cleaning, outlier detection, categorical encoding,
and train/validation/test splitting with stratification.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing.

    Attributes:
        outlier_method: Method for outlier detection ('iqr', 'zscore', 'none')
        outlier_threshold: Threshold for outlier detection (IQR multiplier or z-score)
        cap_outliers: Whether to cap outliers at percentiles vs. remove
        cap_percentiles: Percentiles for capping (lower, upper)
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        stratify_column: Column to use for stratified splitting
        random_state: Random seed for reproducibility
        scale_features: Whether to apply standardization
    """
    outlier_method: str = 'iqr'
    outlier_threshold: float = 1.5
    cap_outliers: bool = True
    cap_percentiles: Tuple[float, float] = (1, 99)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_column: str = 'ship_type'
    random_state: int = 42
    scale_features: bool = False  # Will scale later in models


class DataPreprocessor:
    """Preprocessing pipeline for ship fuel efficiency data.

    Example:
        >>> config = PreprocessingConfig()
        >>> preprocessor = DataPreprocessor(config)
        >>> df = pd.read_csv('data/raw/ship_fuel_efficiency.csv')
        >>> df_clean = preprocessor.clean_data(df)
        >>> df_encoded = preprocessor.encode_categoricals(df_clean)
        >>> train, val, test = preprocessor.split_data(df_encoded)
    """

    def __init__(self, config: PreprocessingConfig):
        """Initialize preprocessor with configuration.

        Args:
            config: PreprocessingConfig object
        """
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame

        Example:
            >>> df_clean = preprocessor.clean_data(df)
            >>> print(f"Removed {len(df) - len(df_clean)} rows")
        """
        df = df.copy()

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"⚠ Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

            # Drop rows with missing target
            if 'fuel_consumption' in df.columns:
                df = df.dropna(subset=['fuel_consumption'])

            # Impute missing engine_efficiency with median by ship_type
            if 'engine_efficiency' in df.columns and df['engine_efficiency'].isnull().any():
                df['engine_efficiency'] = df.groupby('ship_type')['engine_efficiency'].transform(
                    lambda x: x.fillna(x.median())
                )

        # Handle outliers
        if self.config.outlier_method != 'none':
            df = self._handle_outliers(df)

        # Validate data ranges
        df = self._validate_ranges(df)

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using specified method.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        numerical_cols = ['distance', 'fuel_consumption', 'CO2_emissions', 'engine_efficiency']

        for col in numerical_cols:
            if col not in df.columns:
                continue

            if self.config.outlier_method == 'iqr':
                outliers = self._detect_outliers_iqr(df[col], self.config.outlier_threshold)
            elif self.config.outlier_method == 'zscore':
                outliers = self._detect_outliers_zscore(df[col], self.config.outlier_threshold)
            else:
                continue

            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"  {col}: {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)")

                if self.config.cap_outliers:
                    # Cap at percentiles
                    lower = df[col].quantile(self.config.cap_percentiles[0] / 100)
                    upper = df[col].quantile(self.config.cap_percentiles[1] / 100)
                    df[col] = df[col].clip(lower, upper)
                else:
                    # Remove outliers
                    df = df[~outliers]

        return df

    @staticmethod
    def _detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method.

        Args:
            series: Input series
            multiplier: IQR multiplier

        Returns:
            Boolean series indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)

    @staticmethod
    def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method.

        Args:
            series: Input series
            threshold: Z-score threshold

        Returns:
            Boolean series indicating outliers
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    @staticmethod
    def _validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges and remove invalid entries.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with invalid entries removed
        """
        df = df.copy()
        initial_len = len(df)

        # Fuel consumption must be positive
        if 'fuel_consumption' in df.columns:
            df = df[df['fuel_consumption'] > 0]

        # Distance must be positive
        if 'distance' in df.columns:
            df = df[df['distance'] > 0]

        # Engine efficiency between 0-100
        if 'engine_efficiency' in df.columns:
            df = df[(df['engine_efficiency'] > 0) & (df['engine_efficiency'] <= 100)]

        removed = initial_len - len(df)
        if removed > 0:
            print(f"⚠ Removed {removed} rows with invalid ranges")

        return df

    def encode_categoricals(self, df: pd.DataFrame,
                           fit: bool = True) -> pd.DataFrame:
        """Encode categorical features for modeling.

        Encoding strategy:
        - ship_type: One-hot (4 categories)
        - fuel_type: Binary (HFO=1, Diesel=0)
        - weather_conditions: Ordinal (Calm=0, Moderate=1, Stormy=2)
        - route_id: One-hot (4 categories)
        - month: Cyclical sin/cos transformation
        - ship_id: Drop (too many unique values)

        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for train, False for val/test)

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        # Drop ship_id (high cardinality, low value)
        if 'ship_id' in df.columns:
            df = df.drop(columns=['ship_id'])

        # Binary encode fuel_type
        if 'fuel_type' in df.columns:
            df['fuel_type_hfo'] = (df['fuel_type'] == 'HFO').astype(int)
            df = df.drop(columns=['fuel_type'])

        # Ordinal encode weather_conditions
        if 'weather_conditions' in df.columns:
            weather_map = {'Calm': 0, 'Moderate': 1, 'Stormy': 2}
            df['weather_ordinal'] = df['weather_conditions'].map(weather_map)
            df = df.drop(columns=['weather_conditions'])

        # One-hot encode ship_type
        if 'ship_type' in df.columns:
            ship_type_dummies = pd.get_dummies(df['ship_type'], prefix='ship_type')
            df = pd.concat([df, ship_type_dummies], axis=1)
            df = df.drop(columns=['ship_type'])

        # One-hot encode route_id
        if 'route_id' in df.columns:
            route_dummies = pd.get_dummies(df['route_id'], prefix='route')
            df = pd.concat([df, route_dummies], axis=1)
            df = df.drop(columns=['route_id'])

        # Cyclical encode month
        if 'month' in df.columns:
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df['month_num'] = df['month'].map(month_map)
            df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
            df = df.drop(columns=['month', 'month_num'])

        return df

    def split_data(self, df: pd.DataFrame,
                   return_indices: bool = False) -> Tuple[pd.DataFrame, ...]:
        """Split data into train/validation/test sets with stratification.

        Args:
            df: Input DataFrame
            return_indices: Whether to return indices instead of DataFrames

        Returns:
            Tuple of (train, val, test) DataFrames or indices

        Example:
            >>> train, val, test = preprocessor.split_data(df)
            >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        """
        # First split: train+val vs test
        stratify_col = df[self.config.stratify_column] if self.config.stratify_column in df.columns else None

        train_val, test = train_test_split(
            df,
            test_size=self.config.test_ratio,
            random_state=self.config.random_state,
            stratify=stratify_col
        )

        # Second split: train vs val
        val_ratio_adjusted = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        stratify_col_trainval = (train_val[self.config.stratify_column]
                                 if self.config.stratify_column in train_val.columns else None)

        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=self.config.random_state,
            stratify=stratify_col_trainval
        )

        print(f"✓ Data split complete:")
        print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")

        if return_indices:
            return train.index, val.index, test.index
        return train, val, test

    def fit_scaler(self, df: pd.DataFrame, exclude_cols: list = None) -> None:
        """Fit StandardScaler on numerical features.

        Args:
            df: Training DataFrame
            exclude_cols: Columns to exclude from scaling (e.g., target)
        """
        if exclude_cols is None:
            exclude_cols = ['fuel_consumption', 'CO2_emissions']

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]

        self.scaler = StandardScaler()
        self.scaler.fit(df[cols_to_scale])

    def transform_features(self, df: pd.DataFrame,
                          exclude_cols: list = None) -> pd.DataFrame:
        """Transform features using fitted scaler.

        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from scaling

        Returns:
            DataFrame with scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")

        df = df.copy()
        if exclude_cols is None:
            exclude_cols = ['fuel_consumption', 'CO2_emissions']

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]

        df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df

    def get_feature_names(self, df: pd.DataFrame,
                         exclude_target: bool = True) -> list:
        """Get list of feature names after encoding.

        Args:
            df: Encoded DataFrame
            exclude_target: Whether to exclude target variable

        Returns:
            List of feature column names
        """
        exclude_cols = ['fuel_consumption', 'CO2_emissions'] if exclude_target else []
        return [col for col in df.columns if col not in exclude_cols]


def main():
    """Example usage of preprocessing pipeline."""
    # Load data
    df = pd.read_csv('data/raw/ship_fuel_efficiency.csv')
    print(f"Loaded {len(df)} observations\n")

    # Initialize preprocessor
    config = PreprocessingConfig(
        outlier_method='iqr',
        cap_outliers=True,
        cap_percentiles=(1, 99)
    )
    preprocessor = DataPreprocessor(config)

    # Clean data
    print("Cleaning data...")
    df_clean = preprocessor.clean_data(df)
    print(f"After cleaning: {len(df_clean)} observations\n")

    # Encode categoricals
    print("Encoding categorical features...")
    df_encoded = preprocessor.encode_categoricals(df_clean)
    print(f"Features after encoding: {len(df_encoded.columns)}\n")

    # Split data
    print("Splitting data...")
    train, val, test = preprocessor.split_data(df_encoded)

    # Save splits
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    print("\n✓ Splits saved to data/processed/")

    # Print feature summary
    features = preprocessor.get_feature_names(train)
    print(f"\nFeatures for modeling ({len(features)}):")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")


if __name__ == "__main__":
    main()
