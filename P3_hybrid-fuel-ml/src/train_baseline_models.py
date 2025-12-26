"""Train baseline ML models for fuel consumption prediction.

This script trains and evaluates multiple baseline models:
- Ridge Regression (linear baseline)
- XGBoost (gradient boosting)

Generates comparison metrics and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append('.')

from src.feature_engineering import FeatureEngineer
from src.models.linear_model import LinearFuelPredictor
from src.models.xgboost_model import XGBoostFuelPredictor
from src.evaluation import ModelEvaluator


def load_data():
    """Load train/val/test splits."""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    return train, val, test


def prepare_features(train, val, test):
    """Apply feature engineering to all splits."""
    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    # Initialize feature engineer
    fe = FeatureEngineer()
    fe.fit(train)

    # Create enriched features
    train_enriched = fe.create_all_features(train)
    val_enriched = fe.create_all_features(val)
    test_enriched = fe.create_all_features(test)

    # Get feature names for modeling
    feature_names = fe.get_feature_names(train_enriched,
                                         exclude_target=True,
                                         exclude_leakage=True)

    print(f"Features for modeling: {len(feature_names)}")

    return train_enriched, val_enriched, test_enriched, feature_names


def train_ridge_model(X_train, y_train, X_val, y_val, feature_names):
    """Train Ridge regression model."""
    print("\n" + "=" * 80)
    print("TRAINING: Ridge Regression")
    print("=" * 80)

    # Train model
    model = LinearFuelPredictor(alpha=1.0, model_type='ridge')
    model.train(X_train, y_train, feature_names=feature_names)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Evaluate
    evaluator = ModelEvaluator()

    train_metrics = evaluator.compute_metrics(y_train.values, y_train_pred, "Ridge (Train)")
    val_metrics = evaluator.compute_metrics(y_val.values, y_val_pred, "Ridge (Val)")

    evaluator.print_metrics(val_metrics)

    return model, y_val_pred, val_metrics


def train_xgboost_model(X_train, y_train, X_val, y_val, feature_names, tune=True):
    """Train XGBoost model with optional hyperparameter tuning."""
    print("\n" + "=" * 80)
    print("TRAINING: XGBoost")
    print("=" * 80)

    model = XGBoostFuelPredictor()

    if tune:
        # Hyperparameter tuning (reduced iterations for speed)
        print("\nTuning hyperparameters...")
        best_params = model.tune_hyperparameters(
            X_train, y_train,
            n_iter=15,  # Reduced for speed
            cv=3,
            verbose=0
        )

        # Train with best parameters
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
    else:
        # Train with default parameters
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Evaluate
    evaluator = ModelEvaluator()

    train_metrics = evaluator.compute_metrics(y_train.values, y_train_pred, "XGBoost (Train)")
    val_metrics = evaluator.compute_metrics(y_val.values, y_val_pred, "XGBoost (Val)")

    evaluator.print_metrics(val_metrics)

    return model, y_val_pred, val_metrics


def generate_visualizations(models_dict, y_val, predictions_dict, output_dir):
    """Generate comparison visualizations."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator()

    # 1. Predicted vs Actual plots for each model
    for model_name, y_pred in predictions_dict.items():
        evaluator.plot_predictions(
            y_val.values, y_pred,
            title=f'{model_name}: Predicted vs Actual',
            save_path=output_dir / f'{model_name.lower()}_predictions.png'
        )

        evaluator.plot_residuals(
            y_val.values, y_pred,
            title=f'{model_name}: Residual Analysis',
            save_path=output_dir / f'{model_name.lower()}_residuals.png'
        )

    # 2. Feature importance plots
    for model_name, model in models_dict.items():
        importance_df = model.get_feature_importance()
        evaluator.plot_feature_importance(
            importance_df['feature'].tolist(),
            importance_df['importance' if 'importance' in importance_df.columns else 'abs_coefficient'].values,
            title=f'{model_name}: Feature Importance (Top 15)',
            top_n=15,
            save_path=output_dir / f'{model_name.lower()}_importance.png'
        )


def main():
    """Main training pipeline."""
    print("\n" + "#" * 80)
    print("#" + " " * 25 + "MVP-2: BASELINE ML MODELS" + " " * 26 + "#")
    print("#" * 80 + "\n")

    # Output directories
    output_dir = Path('outputs/baseline_models')
    models_dir = Path('models/trained')
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train, val, test = load_data()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Feature engineering
    train_enriched, val_enriched, test_enriched, feature_names = prepare_features(train, val, test)

    # Prepare X and y
    X_train = train_enriched[feature_names]
    y_train = train_enriched['fuel_consumption']

    X_val = val_enriched[feature_names]
    y_val = val_enriched['fuel_consumption']

    X_test = test_enriched[feature_names]
    y_test = test_enriched['fuel_consumption']

    # Store models and predictions
    models = {}
    predictions = {}
    results = {}

    # Train Ridge Regression
    ridge_model, ridge_pred, ridge_metrics = train_ridge_model(
        X_train, y_train, X_val, y_val, feature_names
    )
    models['Ridge'] = ridge_model
    predictions['Ridge'] = ridge_pred
    results['Ridge'] = ridge_metrics

    # Train XGBoost
    xgb_model, xgb_pred, xgb_metrics = train_xgboost_model(
        X_train, y_train, X_val, y_val, feature_names, tune=True
    )
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    results['XGBoost'] = xgb_metrics

    # Generate visualizations
    generate_visualizations(models, y_val, predictions, output_dir)

    # Create comparison table
    evaluator = ModelEvaluator()

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Raw results for comparison
    raw_results = {
        'Ridge': evaluator.compute_metrics(y_val.values, predictions['Ridge'], 'Ridge'),
        'XGBoost': evaluator.compute_metrics(y_val.values, predictions['XGBoost'], 'XGBoost')
    }

    comparison_df = evaluator.create_comparison_table(
        raw_results,
        save_path=output_dir / 'model_comparison.csv'
    )
    print("\n" + comparison_df.to_string())

    # Model comparison bar charts
    evaluator.plot_model_comparison(
        raw_results, metric='R2',
        save_path=output_dir / 'comparison_r2.png'
    )
    evaluator.plot_model_comparison(
        raw_results, metric='RMSE',
        save_path=output_dir / 'comparison_rmse.png'
    )

    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    ridge_model.save(str(models_dir / 'ridge_model.pkl'))
    xgb_model.save(str(models_dir / 'xgboost_model.pkl'))

    # Save best hyperparameters
    if xgb_model.best_params:
        with open(output_dir / 'xgboost_best_params.json', 'w') as f:
            json.dump(xgb_model.best_params, f, indent=2)
        print(f"✓ XGBoost best params saved: {output_dir / 'xgboost_best_params.json'}")

    # Final test set evaluation (only for best model)
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    # Determine best model based on validation R²
    best_model_name = max(raw_results, key=lambda x: raw_results[x]['R2'])
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")

    y_test_pred = best_model.predict(X_test)
    test_metrics = evaluator.compute_metrics(y_test.values, y_test_pred, f"{best_model_name} (Test)")
    evaluator.print_metrics(test_metrics)

    # Test set visualization
    evaluator.plot_predictions(
        y_test.values, y_test_pred,
        title=f'{best_model_name}: Test Set Predictions',
        save_path=output_dir / f'{best_model_name.lower()}_test_predictions.png'
    )

    # Summary
    print("\n" + "#" * 80)
    print("#" + " " * 30 + "SUMMARY" + " " * 30 + "#")
    print("#" * 80)
    print(f"\nBest Model: {best_model_name}")
    print(f"Validation R²: {raw_results[best_model_name]['R2']:.4f}")
    print(f"Test R²: {test_metrics['R2']:.4f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.2f} tonnes")
    print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")

    # Check acceptance criteria
    target_r2 = 0.75
    print(f"\n{'✓' if test_metrics['R2'] >= target_r2 else '✗'} Target R² ≥ {target_r2}: {test_metrics['R2']:.4f}")

    print(f"\nModels saved to: {models_dir}/")
    print(f"Visualizations saved to: {output_dir}/")

    print("\n" + "#" * 80)
    print("#" + " " * 25 + "MVP-2 COMPLETE" + " " * 27 + "#")
    print("#" * 80 + "\n")

    return results


if __name__ == "__main__":
    results = main()
