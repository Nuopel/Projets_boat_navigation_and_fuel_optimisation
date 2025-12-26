"""Calibration analysis for uncertainty quantification.

Provides metrics and visualizations to assess if prediction
intervals are well-calibrated (e.g., 90% CI contains 90% of points).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict


class CalibrationAnalyzer:
    """Analyze calibration of prediction intervals.

    A well-calibrated model should have:
    - 90% CI containing approximately 90% of true values
    - Coverage consistent across different conditions
    """

    @staticmethod
    def compute_coverage(y_true: np.ndarray,
                        lower_bound: np.ndarray,
                        upper_bound: np.ndarray) -> float:
        """Compute Prediction Interval Coverage Probability (PICP).

        Args:
            y_true: True target values
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval

        Returns:
            Coverage as fraction (0-1)
        """
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        return np.mean(within_interval)

    @staticmethod
    def compute_sharpness(lower_bound: np.ndarray,
                         upper_bound: np.ndarray) -> float:
        """Compute Mean Prediction Interval Width (MPIW).

        Smaller is better (more precise) but only if well-calibrated.

        Args:
            lower_bound: Lower bounds
            upper_bound: Upper bounds

        Returns:
            Mean interval width
        """
        return np.mean(upper_bound - lower_bound)

    @staticmethod
    def compute_cwc(y_true: np.ndarray,
                   lower_bound: np.ndarray,
                   upper_bound: np.ndarray,
                   confidence: float = 0.90,
                   eta: float = 50) -> float:
        """Compute Coverage Width-based Criterion (CWC).

        Balances coverage and sharpness with penalty for undercoverage.

        Args:
            y_true: True values
            lower_bound: Lower bounds
            upper_bound: Upper bounds
            confidence: Nominal confidence level
            eta: Penalty coefficient for undercoverage

        Returns:
            CWC score (lower is better)
        """
        picp = CalibrationAnalyzer.compute_coverage(y_true, lower_bound, upper_bound)
        mpiw = CalibrationAnalyzer.compute_sharpness(lower_bound, upper_bound)

        # Normalize MPIW by target range
        target_range = y_true.max() - y_true.min()
        nmpiw = mpiw / target_range if target_range > 0 else mpiw

        # Penalty for undercoverage
        if picp < confidence:
            gamma = np.exp(-eta * (picp - confidence))
        else:
            gamma = 1.0

        return nmpiw * gamma

    @staticmethod
    def compute_all_metrics(y_true: np.ndarray,
                           predictions: np.ndarray,
                           lower_bound: np.ndarray,
                           upper_bound: np.ndarray,
                           confidence: float = 0.90) -> Dict[str, float]:
        """Compute all calibration metrics.

        Args:
            y_true: True values
            predictions: Point predictions
            lower_bound: Lower bounds
            upper_bound: Upper bounds
            confidence: Nominal confidence level

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        picp = CalibrationAnalyzer.compute_coverage(y_true, lower_bound, upper_bound)
        mpiw = CalibrationAnalyzer.compute_sharpness(lower_bound, upper_bound)
        cwc = CalibrationAnalyzer.compute_cwc(y_true, lower_bound, upper_bound, confidence)

        # Calibration error (how far from nominal)
        calibration_error = abs(picp - confidence)

        # Point prediction metrics
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        return {
            'PICP': picp,
            'MPIW': mpiw,
            'CWC': cwc,
            'Calibration_Error': calibration_error,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Nominal_Confidence': confidence
        }

    @staticmethod
    def plot_calibration_scatter(y_true: np.ndarray,
                                predictions: np.ndarray,
                                lower_bound: np.ndarray,
                                upper_bound: np.ndarray,
                                title: str = 'Prediction Intervals',
                                save_path: Optional[Path] = None) -> None:
        """Scatter plot with error bars showing prediction intervals.

        Args:
            y_true: True values
            predictions: Point predictions
            lower_bound: Lower bounds
            upper_bound: Upper bounds
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Determine which points are within interval
        within = (y_true >= lower_bound) & (y_true <= upper_bound)

        # Plot points within interval (green)
        ax.errorbar(
            y_true[within], predictions[within],
            yerr=[predictions[within] - lower_bound[within],
                  upper_bound[within] - predictions[within]],
            fmt='o', color='green', alpha=0.5, ecolor='green',
            elinewidth=1, capsize=2, markersize=4,
            label=f'Within CI ({within.sum()})'
        )

        # Plot points outside interval (red)
        outside = ~within
        if outside.sum() > 0:
            ax.errorbar(
                y_true[outside], predictions[outside],
                yerr=[predictions[outside] - lower_bound[outside],
                      upper_bound[outside] - predictions[outside]],
                fmt='o', color='red', alpha=0.5, ecolor='red',
                elinewidth=1, capsize=2, markersize=4,
                label=f'Outside CI ({outside.sum()})'
            )

        # Perfect prediction line
        min_val = min(y_true.min(), predictions.min())
        max_val = max(y_true.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'k--', linewidth=2, label='Perfect Prediction')

        # Coverage annotation
        coverage = within.mean()
        ax.text(0.05, 0.95, f'Coverage: {coverage:.1%}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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
    def plot_interval_width_distribution(lower_bound: np.ndarray,
                                         upper_bound: np.ndarray,
                                         title: str = 'Interval Width Distribution',
                                         save_path: Optional[Path] = None) -> None:
        """Histogram of prediction interval widths.

        Args:
            lower_bound: Lower bounds
            upper_bound: Upper bounds
            title: Plot title
            save_path: Path to save figure
        """
        widths = upper_bound - lower_bound

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(widths, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(widths.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {widths.mean():.2f}')
        ax.axvline(np.median(widths), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(widths):.2f}')

        ax.set_xlabel('Interval Width (tonnes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_coverage_by_category(y_true: np.ndarray,
                                  lower_bound: np.ndarray,
                                  upper_bound: np.ndarray,
                                  categories: np.ndarray,
                                  category_name: str = 'Category',
                                  save_path: Optional[Path] = None) -> None:
        """Bar chart of coverage by category (e.g., weather).

        Args:
            y_true: True values
            lower_bound: Lower bounds
            upper_bound: Upper bounds
            categories: Category labels for each sample
            category_name: Name of category for plot
            save_path: Path to save figure
        """
        unique_categories = np.unique(categories)

        coverages = []
        counts = []
        for cat in unique_categories:
            mask = categories == cat
            within = ((y_true[mask] >= lower_bound[mask]) &
                     (y_true[mask] <= upper_bound[mask]))
            coverages.append(within.mean())
            counts.append(mask.sum())

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#98D8C8', '#FFD700', '#FF6B6B'][:len(unique_categories)]
        bars = ax.bar(range(len(unique_categories)), coverages,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}\n(n={count})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Reference line for nominal coverage
        ax.axhline(0.9, color='red', linestyle='--', linewidth=2,
                  label='90% Target')

        ax.set_xticks(range(len(unique_categories)))
        ax.set_xticklabels(unique_categories, fontsize=11)
        ax.set_ylabel('Coverage', fontsize=12, fontweight='bold')
        ax.set_xlabel(category_name, fontsize=12, fontweight='bold')
        ax.set_title(f'Coverage by {category_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def print_calibration_summary(metrics: Dict[str, float],
                                 model_name: str = "Model") -> None:
        """Print calibration metrics summary.

        Args:
            metrics: Metrics dictionary from compute_all_metrics()
            model_name: Name of model
        """
        print(f"\n{'='*60}")
        print(f"CALIBRATION: {model_name}")
        print(f"{'='*60}")
        print(f"  Coverage (PICP):     {metrics['PICP']:.2%}")
        print(f"  Nominal Confidence:  {metrics['Nominal_Confidence']:.2%}")
        print(f"  Calibration Error:   {metrics['Calibration_Error']:.2%}")
        print(f"  Mean Width (MPIW):   {metrics['MPIW']:.2f} tonnes")
        print(f"  CWC Score:           {metrics['CWC']:.4f}")
        print(f"  Point RMSE:          {metrics['RMSE']:.2f} tonnes")
        print(f"  Point R²:            {metrics['R2']:.4f}")
        print(f"{'='*60}\n")


def main():
    """Example usage of calibration analysis."""
    print("Calibration analysis module loaded")
    print("Use CalibrationAnalyzer for uncertainty evaluation")


if __name__ == "__main__":
    main()
