"""Train uncertainty quantification models.

This script trains and evaluates:
- Quantile Regression (q05, q50, q95)
- Bootstrap Ensemble (30 models)

Evaluates calibration and generates visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')

from src.feature_engineering import FeatureEngineer
from src.models.uncertainty import QuantileRegressionUncertainty, BootstrapUncertainty
from src.calibration import CalibrationAnalyzer
from src.evaluation import ModelEvaluator


def load_data():
    """Load train/val/test splits."""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    return train, val, test


def prepare_features(train, val, test):
    """Apply feature engineering."""
    fe = FeatureEngineer()
    fe.fit(train)

    train_enriched = fe.create_all_features(train)
    val_enriched = fe.create_all_features(val)
    test_enriched = fe.create_all_features(test)

    feature_names = fe.get_feature_names(train_enriched,
                                         exclude_target=True,
                                         exclude_leakage=True)

    return train_enriched, val_enriched, test_enriched, feature_names


def train_quantile_model(X_train, y_train, feature_names):
    """Train quantile regression model."""
    print("\n" + "=" * 80)
    print("TRAINING: Quantile Regression")
    print("=" * 80)

    model = QuantileRegressionUncertainty(quantiles=(0.05, 0.5, 0.95))
    model.train(X_train[feature_names], y_train, feature_names=feature_names)

    return model


def train_bootstrap_model(X_train, y_train, feature_names):
    """Train bootstrap ensemble model."""
    print("\n" + "=" * 80)
    print("TRAINING: Bootstrap Ensemble")
    print("=" * 80)

    model = BootstrapUncertainty(n_bootstrap=30)
    model.train(X_train[feature_names], y_train, feature_names=feature_names)

    return model


def evaluate_model(model, X, y, model_name, output_dir, feature_names):
    """Evaluate uncertainty model and generate visualizations."""
    print(f"\nEvaluating {model_name}...")

    # Get predictions with intervals
    predictions, lower, upper = model.predict_with_intervals(
        X[feature_names], confidence=0.90
    )

    # Compute calibration metrics
    analyzer = CalibrationAnalyzer()
    metrics = analyzer.compute_all_metrics(
        y.values, predictions, lower, upper, confidence=0.90
    )

    analyzer.print_calibration_summary(metrics, model_name)

    # Generate visualizations
    output_dir = Path(output_dir)

    # 1. Calibration scatter plot
    analyzer.plot_calibration_scatter(
        y.values, predictions, lower, upper,
        title=f'{model_name}: Prediction Intervals (90% CI)',
        save_path=output_dir / f'{model_name.lower().replace(" ", "_")}_calibration.png'
    )

    # 2. Interval width distribution
    analyzer.plot_interval_width_distribution(
        lower, upper,
        title=f'{model_name}: Interval Width Distribution',
        save_path=output_dir / f'{model_name.lower().replace(" ", "_")}_widths.png'
    )

    return metrics, predictions, lower, upper


def main():
    """Main training pipeline for uncertainty quantification."""
    print("\n" + "#" * 80)
    print("#" + " " * 18 + "MVP-4: UNCERTAINTY QUANTIFICATION" + " " * 18 + "#")
    print("#" * 80 + "\n")

    # Output directories
    output_dir = Path('outputs/uncertainty')
    models_dir = Path('models/trained')
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train, val, test = load_data()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Feature engineering
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    train_enriched, val_enriched, test_enriched, feature_names = prepare_features(train, val, test)
    print(f"Features: {len(feature_names)}")

    # Prepare X and y
    X_train = train_enriched
    y_train = train_enriched['fuel_consumption']

    X_val = val_enriched
    y_val = val_enriched['fuel_consumption']

    X_test = test_enriched
    y_test = test_enriched['fuel_consumption']

    # Train models
    quantile_model = train_quantile_model(X_train, y_train, feature_names)
    bootstrap_model = train_bootstrap_model(X_train, y_train, feature_names)

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)

    quantile_metrics_val, q_pred_val, q_lower_val, q_upper_val = evaluate_model(
        quantile_model, X_val, y_val, "Quantile", output_dir, feature_names
    )

    bootstrap_metrics_val, b_pred_val, b_lower_val, b_upper_val = evaluate_model(
        bootstrap_model, X_val, y_val, "Bootstrap", output_dir, feature_names
    )

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    quantile_metrics_test, q_pred_test, q_lower_test, q_upper_test = evaluate_model(
        quantile_model, X_test, y_test, "Quantile_Test", output_dir, feature_names
    )

    bootstrap_metrics_test, b_pred_test, b_lower_test, b_upper_test = evaluate_model(
        bootstrap_model, X_test, y_test, "Bootstrap_Test", output_dir, feature_names
    )

    # Coverage by weather condition
    print("\n" + "=" * 80)
    print("COVERAGE BY WEATHER CONDITION")
    print("=" * 80)

    # Get weather labels
    weather_map = {0: 'Calm', 1: 'Moderate', 2: 'Stormy'}
    weather_labels = X_test['weather_ordinal'].map(weather_map).values

    CalibrationAnalyzer.plot_coverage_by_category(
        y_test.values, q_lower_test, q_upper_test,
        weather_labels, 'Weather Condition',
        save_path=output_dir / 'coverage_by_weather.png'
    )

    # Comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    comparison_data = {
        'Quantile (Val)': {
            'PICP': f"{quantile_metrics_val['PICP']:.2%}",
            'MPIW': f"{quantile_metrics_val['MPIW']:.2f}",
            'R2': f"{quantile_metrics_val['R2']:.4f}",
            'Cal_Error': f"{quantile_metrics_val['Calibration_Error']:.2%}"
        },
        'Bootstrap (Val)': {
            'PICP': f"{bootstrap_metrics_val['PICP']:.2%}",
            'MPIW': f"{bootstrap_metrics_val['MPIW']:.2f}",
            'R2': f"{bootstrap_metrics_val['R2']:.4f}",
            'Cal_Error': f"{bootstrap_metrics_val['Calibration_Error']:.2%}"
        },
        'Quantile (Test)': {
            'PICP': f"{quantile_metrics_test['PICP']:.2%}",
            'MPIW': f"{quantile_metrics_test['MPIW']:.2f}",
            'R2': f"{quantile_metrics_test['R2']:.4f}",
            'Cal_Error': f"{quantile_metrics_test['Calibration_Error']:.2%}"
        },
        'Bootstrap (Test)': {
            'PICP': f"{bootstrap_metrics_test['PICP']:.2%}",
            'MPIW': f"{bootstrap_metrics_test['MPIW']:.2f}",
            'R2': f"{bootstrap_metrics_test['R2']:.4f}",
            'Cal_Error': f"{bootstrap_metrics_test['Calibration_Error']:.2%}"
        }
    }

    comparison_df = pd.DataFrame(comparison_data).T
    print("\n" + comparison_df.to_string())

    # Save comparison
    comparison_df.to_csv(output_dir / 'uncertainty_comparison.csv')
    print(f"\n✓ Saved: {output_dir / 'uncertainty_comparison.csv'}")

    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    quantile_model.save(str(models_dir / 'quantile_regression.pkl'))
    bootstrap_model.save(str(models_dir / 'bootstrap_ensemble.pkl'))

    # Sample predictions table
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS WITH INTERVALS")
    print("=" * 80)

    sample_indices = [0, 50, 100, 150, 200]
    sample_data = []

    for i in sample_indices:
        if i < len(y_test):
            sample_data.append({
                'Actual': f"{y_test.iloc[i]:.0f}",
                'Predicted': f"{q_pred_test[i]:.0f}",
                'Lower (5%)': f"{q_lower_test[i]:.0f}",
                'Upper (95%)': f"{q_upper_test[i]:.0f}",
                'Width': f"{q_upper_test[i] - q_lower_test[i]:.0f}",
                'In CI': '✓' if q_lower_test[i] <= y_test.iloc[i] <= q_upper_test[i] else '✗'
            })

    sample_df = pd.DataFrame(sample_data)
    print("\n" + sample_df.to_string())

    # Summary
    print("\n" + "#" * 80)
    print("#" + " " * 30 + "SUMMARY" + " " * 30 + "#")
    print("#" * 80)

    # Determine best model
    best_model = "Quantile" if quantile_metrics_test['Calibration_Error'] < bootstrap_metrics_test['Calibration_Error'] else "Bootstrap"

    print(f"\nBest Model: {best_model} Regression")
    print(f"Test Coverage (PICP): {quantile_metrics_test['PICP']:.2%}")
    print(f"Target Coverage: 90%")
    print(f"Calibration Error: {quantile_metrics_test['Calibration_Error']:.2%}")
    print(f"Mean Interval Width: {quantile_metrics_test['MPIW']:.2f} tonnes")

    # Check acceptance criteria
    target_picp_low = 0.85
    target_picp_high = 0.92
    actual_picp = quantile_metrics_test['PICP']

    if target_picp_low <= actual_picp <= target_picp_high:
        print(f"\n✓ Calibration target met: {target_picp_low:.0%} ≤ {actual_picp:.0%} ≤ {target_picp_high:.0%}")
    else:
        print(f"\n⚠ Calibration outside target: {actual_picp:.0%} not in [{target_picp_low:.0%}, {target_picp_high:.0%}]")

    print(f"\nModels saved to: {models_dir}/")
    print(f"Visualizations saved to: {output_dir}/")

    print("\n" + "#" * 80)
    print("#" + " " * 25 + "MVP-4 COMPLETE" + " " * 27 + "#")
    print("#" * 80 + "\n")

    return {
        'quantile_val': quantile_metrics_val,
        'bootstrap_val': bootstrap_metrics_val,
        'quantile_test': quantile_metrics_test,
        'bootstrap_test': bootstrap_metrics_test
    }


if __name__ == "__main__":
    results = main()
