"""Data profiling utility for ship fuel efficiency dataset."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the ship fuel efficiency dataset.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(filepath)


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data profile.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with profiling statistics
    """
    profile = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

    # Numerical statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        profile['numerical_stats'] = df[numerical_cols].describe().to_dict()

    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        profile['categorical_stats'] = {
            col: df[col].value_counts().to_dict()
            for col in categorical_cols
        }
        profile['categorical_unique'] = {
            col: df[col].nunique()
            for col in categorical_cols
        }

    return profile


def detect_outliers_iqr(df: pd.DataFrame, column: str,
                        multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method.

    Args:
        df: Input DataFrame
        column: Column name to check
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (df[column] < lower_bound) | (df[column] > upper_bound)


def print_profile_summary(profile: Dict[str, Any]) -> None:
    """Print human-readable profile summary.

    Args:
        profile: Profile dictionary from profile_dataset()
    """
    print("=" * 80)
    print("DATASET PROFILE SUMMARY")
    print("=" * 80)
    print(f"\nShape: {profile['shape'][0]} rows × {profile['shape'][1]} columns")
    print(f"Memory usage: {profile['memory_usage_mb']:.2f} MB")
    print(f"Duplicates: {profile['duplicates']} rows")

    print("\n" + "-" * 80)
    print("MISSING VALUES")
    print("-" * 80)
    for col, pct in profile['missing_percentage'].items():
        if pct > 0:
            print(f"  {col}: {pct:.2f}% ({profile['missing_values'][col]} rows)")

    if 'categorical_unique' in profile:
        print("\n" + "-" * 80)
        print("CATEGORICAL FEATURES")
        print("-" * 80)
        for col, unique_count in profile['categorical_unique'].items():
            print(f"  {col}: {unique_count} unique values")
            top_3 = list(profile['categorical_stats'][col].items())[:3]
            for val, count in top_3:
                print(f"    - {val}: {count}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Load and profile the dataset
    df = load_dataset('data/raw/ship_fuel_efficiency.csv')
    profile = profile_dataset(df)
    print_profile_summary(profile)

    # Check for outliers in numerical columns
    print("\nOUTLIER DETECTION (IQR method)")
    print("-" * 80)
    numerical_cols = ['distance', 'fuel_consumption', 'CO2_emissions', 'engine_efficiency']
    for col in numerical_cols:
        if col in df.columns:
            outliers = detect_outliers_iqr(df, col)
            print(f"  {col}: {outliers.sum()} outliers ({outliers.sum()/len(df)*100:.2f}%)")
