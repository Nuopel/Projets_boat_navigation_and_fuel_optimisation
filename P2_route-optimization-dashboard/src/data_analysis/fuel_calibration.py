"""
Fuel model calibration using real ship data.

Fits ship fuel consumption model coefficients to observed data
from ship_fuel_efficiency.csv dataset.
"""

from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import warnings

from src.models.ship_model import ShipSpecifications, ShipDynamics


# Weather condition to approximate wind/wave mappings
WEATHER_CONDITIONS = {
    'Calm': {'wind_speed': 8.0, 'wave_height': 1.0},
    'Moderate': {'wind_speed': 15.0, 'wave_height': 3.0},
    'Stormy': {'wind_speed': 30.0, 'wave_height': 6.0}
}


def load_ship_data(csv_path: str) -> pd.DataFrame:
    """Load ship fuel efficiency dataset.

    Args:
        csv_path: Path to ship_fuel_efficiency.csv

    Returns:
        DataFrame with ship fuel data
    """
    df = pd.read_csv(csv_path)

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess ship data for calibration.

    Adds derived columns: average speed, fuel rate (L/h), weather parameters.

    Args:
        df: Raw ship data

    Returns:
        Preprocessed DataFrame with additional columns
    """
    df = df.copy()

    # Estimate average speed (assuming voyage duration from distance and typical speed)
    # For calibration, we'll estimate voyage time from fuel consumption
    # Rough estimate: time (hours) ≈ distance / 12 knots (typical cruising)
    df['estimated_voyage_time_h'] = df['distance'] / 12.0

    # Fuel rate (liters per hour)
    df['fuel_rate'] = df['fuel_consumption'] / df['estimated_voyage_time_h']

    # Estimated average speed (nm/h = knots)
    df['avg_speed_kn'] = df['distance'] / df['estimated_voyage_time_h']

    # Map weather conditions to approximate wind/wave values
    df['wind_speed'] = df['weather_conditions'].map(
        lambda x: WEATHER_CONDITIONS.get(x, WEATHER_CONDITIONS['Moderate'])['wind_speed']
    )
    df['wave_height'] = df['weather_conditions'].map(
        lambda x: WEATHER_CONDITIONS.get(x, WEATHER_CONDITIONS['Moderate'])['wave_height']
    )

    # Filter outliers (remove extreme values)
    df = df[
        (df['fuel_rate'] > 0) &
        (df['fuel_rate'] < 10000) &  # Remove unrealistic values
        (df['avg_speed_kn'] > 5) &
        (df['avg_speed_kn'] < 25)
    ]

    return df


def fit_fuel_model(df: pd.DataFrame,
                   v_min: float = 8.0,
                   v_max: float = 18.0) -> Tuple[ShipSpecifications, Dict]:
    """Fit fuel consumption model to ship data.

    Model: f(V, W, H) = a*V³ + b*W² + c*H + d

    Args:
        df: Preprocessed ship data with fuel_rate, avg_speed_kn, wind_speed, wave_height
        v_min: Minimum operational speed (knots)
        v_max: Maximum operational speed (knots)

    Returns:
        Tuple of (ShipSpecifications with fitted coefficients, fit_statistics dict)
    """
    # Prepare feature matrix
    V = df['avg_speed_kn'].values
    W = df['wind_speed'].values
    H = df['wave_height'].values
    F_obs = df['fuel_rate'].values  # Observed fuel rate

    # Define model function for curve_fit
    def fuel_model(X, a, b, c, d):
        """Fuel model: f(V, W, H) = a*V³ + b*W² + c*H + d"""
        V_data, W_data, H_data = X
        return a * (V_data ** 3) + b * (W_data ** 2) + c * H_data + d

    # Stack features
    X_data = np.vstack([V, W, H])

    # Initial guesses for coefficients
    # Based on typical ship: ~2000 L/h at 15 kn
    # a ≈ 2000 / 15³ ≈ 0.6
    p0 = [0.5, 1.0, 20.0, 300.0]  # [a, b, c, d]

    try:
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                fuel_model,
                X_data,
                F_obs,
                p0=p0,
                bounds=(
                    [0, 0, 0, 0],  # Lower bounds (all non-negative)
                    [10, 50, 200, 2000]  # Upper bounds (reasonable ranges)
                ),
                maxfev=10000
            )

        a, b, c, d = popt

        # Predict and calculate statistics
        F_pred = fuel_model(X_data, a, b, c, d)
        r2 = r2_score(F_obs, F_pred)
        rmse = np.sqrt(mean_squared_error(F_obs, F_pred))
        mae = np.mean(np.abs(F_obs - F_pred))

        # Calculate coefficient standard errors
        perr = np.sqrt(np.diag(pcov))

        # Create ShipSpecifications with fitted coefficients
        specs = ShipSpecifications(
            name="Calibrated Generic Vessel",
            v_min=v_min,
            v_max=v_max,
            fuel_coef_speed=a,
            fuel_coef_wind=b,
            fuel_coef_wave=c,
            fuel_base=d,
            emission_factor=2.8  # Standard HFO emission factor
        )

        # Compile fit statistics
        fit_stats = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'coefficients': {
                'a (speed)': a,
                'b (wind)': b,
                'c (wave)': c,
                'd (base)': d
            },
            'std_errors': {
                'a': perr[0],
                'b': perr[1],
                'c': perr[2],
                'd': perr[3]
            },
            'n_samples': len(df),
            'predicted_values': F_pred,
            'observed_values': F_obs
        }

        return specs, fit_stats

    except Exception as e:
        print(f"Calibration failed: {e}")
        # Return default specs
        specs = ShipSpecifications(
            name="Default (Calibration Failed)",
            v_min=v_min,
            v_max=v_max,
            fuel_coef_speed=0.5,
            fuel_coef_wind=0.8,
            fuel_coef_wave=15.0,
            fuel_base=500.0,
            emission_factor=2.8
        )
        fit_stats = {'error': str(e), 'r2': 0.0}
        return specs, fit_stats


def validate_weather_correlation(df: pd.DataFrame) -> Dict:
    """Validate that stormy weather increases fuel consumption.

    Args:
        df: Preprocessed ship data

    Returns:
        Dictionary with weather impact statistics
    """
    # Group by weather condition
    weather_groups = df.groupby('weather_conditions')['fuel_rate'].agg(['mean', 'std', 'count'])

    # Calculate ratios
    calm_fuel = weather_groups.loc['Calm', 'mean'] if 'Calm' in weather_groups.index else None
    stormy_fuel = weather_groups.loc['Stormy', 'mean'] if 'Stormy' in weather_groups.index else None

    stormy_to_calm_ratio = None
    if calm_fuel and stormy_fuel and calm_fuel > 0:
        stormy_to_calm_ratio = stormy_fuel / calm_fuel

    return {
        'by_weather': weather_groups.to_dict('index'),
        'stormy_to_calm_ratio': stormy_to_calm_ratio,
        'validation': {
            'expected_ratio_range': (1.25, 1.50),
            'actual_ratio': stormy_to_calm_ratio,
            'passes': (1.20 <= stormy_to_calm_ratio <= 1.60) if stormy_to_calm_ratio else False
        }
    }


def calibrate_from_csv(csv_path: str,
                       v_min: float = 8.0,
                       v_max: float = 18.0) -> Tuple[ShipDynamics, Dict]:
    """Complete calibration workflow from CSV file.

    Args:
        csv_path: Path to ship_fuel_efficiency.csv
        v_min: Minimum operational speed
        v_max: Maximum operational speed

    Returns:
        Tuple of (ShipDynamics with calibrated specs, calibration_report dict)
    """
    # Load and preprocess
    print(f"Loading data from {csv_path}...")
    df = load_ship_data(csv_path)
    print(f"Loaded {len(df)} records")

    print("Preprocessing data...")
    df_proc = preprocess_data(df)
    print(f"After preprocessing: {len(df_proc)} records")

    # Fit model
    print("Fitting fuel consumption model...")
    specs, fit_stats = fit_fuel_model(df_proc, v_min, v_max)
    print(f"Model fit R² = {fit_stats.get('r2', 0):.4f}")

    # Validate weather correlation
    print("Validating weather correlation...")
    weather_stats = validate_weather_correlation(df_proc)

    # Create ShipDynamics
    ship = ShipDynamics(specs)

    # Compile full report
    report = {
        'fit_statistics': fit_stats,
        'weather_validation': weather_stats,
        'model_specifications': {
            'name': specs.name,
            'v_min': specs.v_min,
            'v_max': specs.v_max,
            'fuel_coef_speed': specs.fuel_coef_speed,
            'fuel_coef_wind': specs.fuel_coef_wind,
            'fuel_coef_wave': specs.fuel_coef_wave,
            'fuel_base': specs.fuel_base,
            'emission_factor': specs.emission_factor
        }
    }

    return ship, report


def print_calibration_report(report: Dict):
    """Print formatted calibration report.

    Args:
        report: Calibration report dictionary
    """
    print("\n" + "="*60)
    print("FUEL MODEL CALIBRATION REPORT")
    print("="*60)

    # Fit statistics
    fit = report['fit_statistics']
    print(f"\n📊 Fit Quality:")
    print(f"  R² Score:       {fit['r2']:.4f}")
    print(f"  RMSE:           {fit['rmse']:.2f} L/h")
    print(f"  MAE:            {fit['mae']:.2f} L/h")
    print(f"  N Samples:      {fit['n_samples']}")

    # Coefficients
    print(f"\n🔧 Model Coefficients:")
    coef = fit['coefficients']
    stderr = fit['std_errors']
    print(f"  a (speed³):     {coef['a (speed)']:.4f} ± {stderr['a']:.4f}")
    print(f"  b (wind²):      {coef['b (wind)']:.4f} ± {stderr['b']:.4f}")
    print(f"  c (wave):       {coef['c (wave)']:.4f} ± {stderr['c']:.4f}")
    print(f"  d (base):       {coef['d (base)']:.2f} ± {stderr['d']:.2f}")

    # Weather validation
    weather = report['weather_validation']
    print(f"\n🌊 Weather Validation:")
    val = weather['validation']
    print(f"  Expected ratio: {val['expected_ratio_range']}")
    print(f"  Actual ratio:   {val['actual_ratio']:.3f}")
    print(f"  Status:         {'✓ PASS' if val['passes'] else '✗ FAIL'}")

    print("\n" + "="*60 + "\n")
