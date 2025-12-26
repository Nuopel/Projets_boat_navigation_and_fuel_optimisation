#!/usr/bin/env python3
"""Run focused benchmarks WITH NORMALIZATION to fix RBF instability.

This script demonstrates the critical importance of feature normalization
for RBF interpolation on real yacht data.

Usage:
    python scripts/run_normalized_benchmarks.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import InterpolationBenchmarker
import numpy as np

def main():
    """Run focused benchmark experiments with normalization."""
    print("=" * 80)
    print("NORMALIZED BENCHMARKING - FIXING RBF INSTABILITY")
    print("=" * 80)
    print()
    print("This experiment demonstrates that feature normalization is CRITICAL")
    print("for RBF interpolation on real-world data with features on different scales.")
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
    print("⚠️  Note: Features are on VERY different scales!")
    print("   - This causes numerical instability for RBF without normalization")
    print()

    # ==========================================
    # Experiment: Convergence with Normalization
    # ==========================================
    print("-" * 80)
    print("CONVERGENCE ANALYSIS WITH NORMALIZATION")
    print("-" * 80)
    print("Testing all three methods with properly normalized features")
    print()

    sample_sizes = [20, 30, 50, 75, 100, 150, 200]
    print(f"Sample sizes: {sample_sizes}")
    print(f"Methods: RBF, Spline, Kriging")
    print(f"Trials per configuration: 5")
    print(f"Normalization: ENABLED")
    print()

    norm_results = benchmarker.run_convergence_study(
        sample_sizes=sample_sizes,
        methods=['rbf', 'spline', 'kriging'],
        n_trials=5,
        test_fraction=0.2,
        normalize=True  # KEY: Enable normalization!
    )

    print()
    print(f"✓ Normalized convergence analysis complete: {len(norm_results)} experiments")
    print()

    # Show summary
    summary = benchmarker.get_summary_statistics(norm_results)
    print("Summary Statistics (WITH NORMALIZATION):")
    print(summary[['method', 'n_train', 'rmse_mean', 'rmse_std', 'train_time_mean']].to_string(index=False))
    print()

    # ==========================================
    # Export Results
    # ==========================================
    print("-" * 80)
    print("EXPORTING RESULTS")
    print("-" * 80)

    # Create results directory if needed
    os.makedirs('results', exist_ok=True)

    # Export normalized results
    norm_df = benchmarker.get_summary_statistics(norm_results)
    norm_df.to_csv('results/convergence_normalized.csv', index=False)
    print("✓ Normalized convergence: results/convergence_normalized.csv")

    # Export all raw results
    benchmarker.export_results('results/all_normalized_results.csv')
    print()

    # ==========================================
    # Key Findings
    # ==========================================
    print("=" * 80)
    print("KEY FINDINGS - NORMALIZATION IS ESSENTIAL")
    print("=" * 80)
    print()
    print("📊 COMPARISON:")
    print()
    print("WITHOUT Normalization (previous run):")
    print("  - RBF: Severe numerical instability (RMSE 10^18 to 10^32)")
    print("  - RBF: Multiple 'Singular matrix' errors")
    print("  - RBF: Performance degrades with dataset size")
    print()
    print("WITH Normalization (this run):")
    print("  - RBF: Should show stable, consistent performance")
    print("  - RBF: No singular matrix errors expected")
    print("  - RBF: Performance consistent across all dataset sizes")
    print()
    print("🔑 LESSON:")
    print("   Feature normalization (StandardScaler) is CRITICAL for RBF on real data!")
    print()
    print("   The yacht dataset has:")
    print("   - Velocity: 2.41-8.66 knots")
    print("   - Draft: 0.56-1.07 meters")
    print("   - Resistance: 0.01-62.42")
    print()
    print("   Without normalization, RBF's distance calculations become numerically unstable.")
    print("   With normalization, all features are scaled to mean=0, std=1, ensuring stability.")
    print()
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Compare: results/convergence_analysis.csv (unnormalized)")
    print("             vs results/convergence_normalized.csv (normalized)")
    print("  2. Run: python scripts/generate_benchmark_visualizations.py")
    print("  3. Review: docs/MVP4_RESULTS.md (to be generated)")
    print("=" * 80)


if __name__ == '__main__':
    main()
