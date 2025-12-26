#!/usr/bin/env python3
"""Run comprehensive benchmarking suite for all interpolation methods.

This script executes:
1. Convergence analysis (varying sample sizes)
2. Noise robustness testing (varying SNR levels)
3. Exports results to CSV for analysis

Usage:
    python scripts/run_benchmarks.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import InterpolationBenchmarker
import numpy as np

def main():
    """Run all benchmark experiments."""
    print("=" * 80)
    print("INTERPOLATION METHODS BENCHMARKING SUITE")
    print("=" * 80)
    print()

    # Initialize benchmarker
    benchmarker = InterpolationBenchmarker(random_state=42)

    # Load UCI Yacht data
    print("Loading UCI Yacht Hydrodynamics dataset...")
    X, y = benchmarker.load_yacht_data(
        'yacht_hydro.xls',
        aggregate_duplicates=True
    )
    print(f"✓ Loaded {len(X)} aggregated samples")
    print(f"  - Velocity range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}] knots")
    print(f"  - Draft range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}] meters")
    print(f"  - Resistance range: [{y.min():.2f}, {y.max():.2f}]")
    print()

    # ==========================================
    # Experiment 1: Convergence Analysis
    # ==========================================
    print("-" * 80)
    print("EXPERIMENT 1: Convergence Analysis")
    print("-" * 80)
    print("Testing how accuracy improves with more training data")
    print()

    sample_sizes = [10, 20, 30, 50, 75, 100, 150, 200]
    print(f"Sample sizes: {sample_sizes}")
    print(f"Methods: RBF, Spline, Kriging")
    print(f"Trials per configuration: 5")
    print()

    conv_results = benchmarker.run_convergence_study(
        sample_sizes=sample_sizes,
        methods=['rbf', 'spline', 'kriging'],
        n_trials=5,
        test_fraction=0.2
    )

    print()
    print(f"✓ Convergence analysis complete: {len(conv_results)} experiments")
    print()

    # Show summary
    summary = benchmarker.get_summary_statistics(conv_results)
    print("Summary Statistics:")
    print(summary[['method', 'n_train', 'rmse_mean', 'rmse_std', 'train_time_mean']].to_string(index=False))
    print()

    # ==========================================
    # Experiment 2: Noise Robustness
    # ==========================================
    print("-" * 80)
    print("EXPERIMENT 2: Noise Robustness Testing")
    print("-" * 80)
    print("Testing performance under measurement noise")
    print()

    noise_levels = [40, 30, 20, 15, 10]  # SNR in dB (lower = more noise)
    print(f"SNR levels: {noise_levels} dB")
    print(f"Training samples: 100")
    print(f"Trials per configuration: 5")
    print()

    noise_results = benchmarker.run_noise_robustness_study(
        noise_levels=noise_levels,
        n_train=100,
        methods=['rbf', 'spline', 'kriging'],
        n_trials=5
    )

    print()
    print(f"✓ Noise robustness analysis complete: {len(noise_results)} experiments")
    print()

    # ==========================================
    # Export Results
    # ==========================================
    print("-" * 80)
    print("EXPORTING RESULTS")
    print("-" * 80)

    # Create results directory if needed
    os.makedirs('results', exist_ok=True)

    # Export convergence results
    conv_df = benchmarker.get_summary_statistics(conv_results)
    conv_df.to_csv('results/convergence_analysis.csv', index=False)
    print("✓ Convergence results: results/convergence_analysis.csv")

    # Export noise results
    noise_df = benchmarker.get_summary_statistics(noise_results)
    noise_df.to_csv('results/noise_robustness.csv', index=False)
    print("✓ Noise robustness results: results/noise_robustness.csv")

    # Export all raw results
    benchmarker.export_results('results/all_benchmark_results.csv')
    print()

    # ==========================================
    # Final Summary
    # ==========================================
    print("=" * 80)
    print("BENCHMARKING COMPLETE")
    print("=" * 80)
    print(f"Total experiments run: {len(benchmarker.results)}")
    print(f"Convergence tests: {len(conv_results)}")
    print(f"Noise robustness tests: {len(noise_results)}")
    print()
    print("Results saved to:")
    print("  - results/convergence_analysis.csv")
    print("  - results/noise_robustness.csv")
    print("  - results/all_benchmark_results.csv")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/generate_benchmark_visualizations.py")
    print("  2. Review: docs/MVP4_RESULTS.md (to be generated)")
    print("=" * 80)


if __name__ == '__main__':
    main()
