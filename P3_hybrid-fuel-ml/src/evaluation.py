"""Model evaluation utilities for regression tasks.

This module provides comprehensive evaluation metrics and visualizations
for fuel consumption prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error
)


class ModelEvaluator:
    """Comprehensive evaluation for regression models.

    Provides metrics computation, visualizations, and comparison utilities.

    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.compute_metrics(y_true, y_pred)
        >>> evaluator.plot_predictions(y_true, y_pred, 'outputs/model_name/')
    """

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "Model") -> Dict[str, float]:
        """Compute comprehensive regression metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of model for display

        Returns:
            Dictionary of metric names and values

        Metrics:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - MAPE: Mean Absolute Percentage Error (%)
        - R²: Coefficient of determination
        - Max Error: Maximum absolute error
        """
        metrics = {
            'model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred),
            'Max_Error': max_error(y_true, y_pred)
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float]) -> None:
        """Print metrics in formatted table.

        Args:
            metrics: Dictionary from compute_metrics()
        """
        print(f"\n{'='*60}")
        print(f"MODEL: {metrics['model']}")
        print(f"{'='*60}")
        print(f"  RMSE:      {metrics['RMSE']:,.2f} tonnes")
        print(f"  MAE:       {metrics['MAE']:,.2f} tonnes")
        print(f"  MAPE:      {metrics['MAPE']:.2f}%")
        print(f"  R²:        {metrics['R2']:.4f}")
        print(f"  Max Error: {metrics['Max_Error']:,.2f} tonnes")
        print(f"{'='*60}\n")

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = 'Predicted vs Actual',
                        save_path: Optional[Path] = None) -> None:
        """Scatter plot: predicted vs. actual with perfect prediction line.

        Args:
            y_true: True target values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure (if None, displays)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50,
                  edgecolors='black', linewidth=0.5, color='steelblue')

        # Perfect prediction line (y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, label='Perfect Prediction')

        # Compute R²
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Add metrics text box
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', bbox=props)

        ax.set_xlabel('Actual Fuel Consumption (tonnes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Fuel Consumption (tonnes)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = 'Residual Analysis',
                      save_path: Optional[Path] = None) -> None:
        """Residual plot to check for heteroscedasticity and patterns.

        Args:
            y_true: True target values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, s=50,
                       edgecolors='black', linewidth=0.5, color='coral')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[0].set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
        axes[0].grid(alpha=0.3)

        # Right: Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1].axvline(residuals.mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {residuals.mean():.2f}')
        axes[1].set_xlabel('Residual Value', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_feature_importance(feature_names: list, importances: np.ndarray,
                               title: str = 'Feature Importance',
                               top_n: int = 15,
                               save_path: Optional[Path] = None) -> None:
        """Plot feature importance bar chart.

        Args:
            feature_names: List of feature names
            importances: Feature importance values
            title: Plot title
            top_n: Number of top features to display
            save_path: Path to save figure
        """
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(importances)  # Use absolute for linear models
        }).sort_values('importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.barh(range(len(importance_df)), importance_df['importance'],
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def create_comparison_table(results: Dict[str, Dict[str, float]],
                               save_path: Optional[Path] = None) -> pd.DataFrame:
        """Create comparison table for multiple models.

        Args:
            results: Dict of {model_name: metrics_dict}
            save_path: Path to save CSV

        Returns:
            DataFrame with model comparison

        Example:
            >>> results = {
            ...     'Ridge': evaluator.compute_metrics(y_true, y_pred_ridge),
            ...     'XGBoost': evaluator.compute_metrics(y_true, y_pred_xgb)
            ... }
            >>> df = evaluator.create_comparison_table(results)
        """
        comparison_df = pd.DataFrame(results).T

        # Sort by R² descending
        if 'R2' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('R2', ascending=False)

        # Format for display
        comparison_df['RMSE'] = comparison_df['RMSE'].apply(lambda x: f'{x:,.2f}')
        comparison_df['MAE'] = comparison_df['MAE'].apply(lambda x: f'{x:,.2f}')
        comparison_df['MAPE'] = comparison_df['MAPE'].apply(lambda x: f'{x:.2f}%')
        comparison_df['R2'] = comparison_df['R2'].apply(lambda x: f'{x:.4f}')
        comparison_df['Max_Error'] = comparison_df['Max_Error'].apply(lambda x: f'{x:,.2f}')

        if save_path:
            comparison_df.to_csv(save_path)
            print(f"✓ Saved comparison table: {save_path}")

        return comparison_df

    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict[str, float]],
                             metric: str = 'R2',
                             save_path: Optional[Path] = None) -> None:
        """Bar chart comparing models on a specific metric.

        Args:
            results: Dict of {model_name: metrics_dict}
            metric: Metric to compare ('R2', 'RMSE', 'MAE', 'MAPE')
            save_path: Path to save figure
        """
        models = list(results.keys())
        values = [results[model][metric] for model in models]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(models)]
        bars = ax.bar(models, values, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}' if metric == 'R2' else f'{height:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Rotate x-labels if many models
        if len(models) > 3:
            plt.xticks(rotation=15, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """Example usage of evaluation module."""
    print("Evaluation module loaded successfully")
    print("Use ModelEvaluator class to evaluate your models")


if __name__ == "__main__":
    main()
