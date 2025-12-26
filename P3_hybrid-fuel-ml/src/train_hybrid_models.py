"""Train hybrid Physics-ML models for fuel consumption prediction.

This script trains and evaluates hybrid models:
- Physics Baseline
- Residual Correction Hybrid
- Feature Augmentation Hybrid

Compares against pure ML baseline from MVP-2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append('.')

from src.feature_engineering import FeatureEngineer
from src.models.physics_baseline import PhysicsBasedFuelModel
from src.models.hybrid_model import ResidualCorrectionHybrid, FeatureAugmentationHybrid
from src.models.linear_model import LinearFuelPredictor
from src.evaluation import ModelEvaluator


def load_data():
    """Load train/val/test splits."""
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    return train, val, test


def prepare_features(train, val, test):
    """Apply feature engineering to all splits."""
    fe = FeatureEngineer()
    fe.fit(train)

    train_enriched = fe.create_all_features(train)
    val_enriched = fe.create_all_features(val)
    test_enriched = fe.create_all_features(test)

    feature_names = fe.get_feature_names(train_enriched,
                                         exclude_target=True,
                                         exclude_leakage=True)

    return train_enriched, val_enriched, test_enriched, feature_names


def train_physics_baseline(X_train, y_train, X_val, y_val):
    """Train and evaluate physics baseline model."""
    print("\n" + "=" * 80)
    print("TRAINING: Physics Baseline Model")
    print("=" * 80)

    model = PhysicsBasedFuelModel()
    model.calibrate(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    evaluator = ModelEvaluator()
    train_metrics = evaluator.compute_metrics(y_train.values, y_train_pred, "Physics (Train)")
    val_metrics = evaluator.compute_metrics(y_val.values, y_val_pred, "Physics (Val)")

    evaluator.print_metrics(val_metrics)

    return model, y_val_pred, val_metrics


def train_residual_hybrid(X_train, y_train, X_val, y_val, feature_names):
    """Train residual correction hybrid model."""
    model = ResidualCorrectionHybrid()
    model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    evaluator = ModelEvaluator()
    val_metrics = evaluator.compute_metrics(y_val.values, y_val_pred, "Residual Hybrid (Val)")
    evaluator.print_metrics(val_metrics)

    return model, y_val_pred, val_metrics


def train_feature_hybrid(X_train, y_train, X_val, y_val, feature_names):
    """Train feature augmentation hybrid model."""
    model = FeatureAugmentationHybrid()
    model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    evaluator = ModelEvaluator()
    val_metrics = evaluator.compute_metrics(y_val.values, y_val_pred, "Feature Hybrid (Val)")
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

    # Predicted vs Actual for each model
    for model_name, y_pred in predictions_dict.items():
        evaluator.plot_predictions(
            y_val.values, y_pred,
            title=f'{model_name}: Predicted vs Actual',
            save_path=output_dir / f'{model_name.lower().replace(" ", "_")}_predictions.png'
        )

    # Feature importance for hybrid models
    for model_name, model in models_dict.items():
        if hasattr(model, 'get_feature_importance'):
            try:
                importance_df = model.get_feature_importance()
                evaluator.plot_feature_importance(
                    importance_df['feature'].tolist(),
                    importance_df['importance'].values,
                    title=f'{model_name}: Feature Importance (Top 15)',
                    top_n=15,
                    save_path=output_dir / f'{model_name.lower().replace(" ", "_")}_importance.png'
                )
            except Exception as e:
                print(f"Could not generate importance for {model_name}: {e}")


def main():
    """Main training pipeline for hybrid models."""
    print("\n" + "#" * 80)
    print("#" + " " * 20 + "MVP-3: HYBRID PHYSICS-ML MODELS" + " " * 20 + "#")
    print("#" * 80 + "\n")

    # Output directories
    output_dir = Path('outputs/hybrid_model')
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
    print(f"Features for modeling: {len(feature_names)}")

    # Prepare X and y
    X_train = train_enriched
    y_train = train_enriched['fuel_consumption']

    X_val = val_enriched
    y_val = val_enriched['fuel_consumption']

    X_test = test_enriched
    y_test = test_enriched['fuel_consumption']

    # Store models and predictions
    models = {}
    predictions = {}
    results = {}

    # Load pure ML baseline for comparison
    print("\n" + "=" * 80)
    print("LOADING BASELINE: Ridge Regression (from MVP-2)")
    print("=" * 80)

    ridge_model = LinearFuelPredictor.load('models/trained/ridge_model.pkl')
    ridge_pred = ridge_model.predict(X_val[feature_names])
    ridge_metrics = ModelEvaluator.compute_metrics(y_val.values, ridge_pred, "Ridge (Val)")
    ModelEvaluator.print_metrics(ridge_metrics)

    models['Ridge'] = ridge_model
    predictions['Ridge'] = ridge_pred
    results['Ridge'] = ridge_metrics

    # Train Physics Baseline
    physics_model, physics_pred, physics_metrics = train_physics_baseline(
        X_train, y_train, X_val, y_val
    )
    models['Physics'] = physics_model
    predictions['Physics'] = physics_pred
    results['Physics'] = physics_metrics

    # Train Residual Correction Hybrid
    residual_model, residual_pred, residual_metrics = train_residual_hybrid(
        X_train, y_train, X_val, y_val, feature_names
    )
    models['Residual Hybrid'] = residual_model
    predictions['Residual Hybrid'] = residual_pred
    results['Residual Hybrid'] = residual_metrics

    # Train Feature Augmentation Hybrid
    feature_model, feature_pred, feature_metrics = train_feature_hybrid(
        X_train, y_train, X_val, y_val, feature_names
    )
    models['Feature Hybrid'] = feature_model
    predictions['Feature Hybrid'] = feature_pred
    results['Feature Hybrid'] = feature_metrics

    # Generate visualizations
    generate_visualizations(models, y_val, predictions, output_dir)

    # Create comparison table
    evaluator = ModelEvaluator()

    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 80)

    # Raw results for comparison
    raw_results = {}
    for name in ['Ridge', 'Physics', 'Residual Hybrid', 'Feature Hybrid']:
        raw_results[name] = results[name]

    comparison_df = evaluator.create_comparison_table(
        raw_results,
        save_path=output_dir / 'hybrid_comparison.csv'
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

    # Save hybrid models
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    physics_model.save(str(models_dir / 'physics_baseline.pkl'))
    residual_model.save(str(models_dir / 'hybrid_residual.pkl'))
    feature_model.save(str(models_dir / 'hybrid_feature.pkl'))

    # Determine best model
    best_model_name = max(raw_results, key=lambda x: raw_results[x]['R2'])
    best_model = models[best_model_name]

    # Final test set evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    print(f"\nBest model: {best_model_name}")

    # Test predictions for all models
    test_results = {}

    for name, model in models.items():
        if name == 'Ridge':
            y_test_pred = model.predict(X_test[feature_names])
        elif name == 'Physics':
            y_test_pred = model.predict(X_test)
        else:
            y_test_pred = model.predict(X_test)

        test_metrics = evaluator.compute_metrics(y_test.values, y_test_pred, f"{name} (Test)")
        test_results[name] = test_metrics

        if name == best_model_name:
            evaluator.print_metrics(test_metrics)
            evaluator.plot_predictions(
                y_test.values, y_test_pred,
                title=f'{name}: Test Set Predictions',
                save_path=output_dir / f'{name.lower().replace(" ", "_")}_test_predictions.png'
            )

    # Save test results
    test_comparison = evaluator.create_comparison_table(
        test_results,
        save_path=output_dir / 'test_comparison.csv'
    )
    print("\nTest Set Results:")
    print(test_comparison.to_string())

    # Summary
    print("\n" + "#" * 80)
    print("#" + " " * 30 + "SUMMARY" + " " * 30 + "#")
    print("#" * 80)

    # Calculate improvement
    ridge_rmse = results['Ridge']['RMSE']
    best_rmse = results[best_model_name]['RMSE']
    improvement = (ridge_rmse - best_rmse) / ridge_rmse * 100

    print(f"\nBest Model: {best_model_name}")
    print(f"Validation R²: {results[best_model_name]['R2']:.4f}")
    print(f"Test R²: {test_results[best_model_name]['R2']:.4f}")
    print(f"Test RMSE: {test_results[best_model_name]['RMSE']:.2f} tonnes")

    print(f"\nHybrid vs Ridge Improvement:")
    print(f"  RMSE: {improvement:.2f}% {'improvement' if improvement > 0 else 'regression'}")

    # Physics model contribution
    physics_r2 = results['Physics']['R2']
    print(f"\nPhysics Baseline R²: {physics_r2:.4f}")
    print(f"{'✓' if physics_r2 > 0.5 else '✗'} Physics baseline > 0.5 R²")

    print(f"\nModels saved to: {models_dir}/")
    print(f"Visualizations saved to: {output_dir}/")

    print("\n" + "#" * 80)
    print("#" + " " * 25 + "MVP-3 COMPLETE" + " " * 27 + "#")
    print("#" * 80 + "\n")

    return results, test_results


if __name__ == "__main__":
    results, test_results = main()
