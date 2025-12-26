"""Generate visualizations for MVP-1 and MVP-2 documentation.

This script creates figures demonstrating:
- UCI yacht data distribution
- Derived (V, T) surface
- Sampling strategy comparison
- Synthetic resistance surface
- Noise impact visualization
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from src.data.loader import YachtDataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.synthetic import SyntheticSurfaceGenerator

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

def create_mvp1_visualizations():
    """Generate MVP-1 visualizations."""
    print("Generating MVP-1 visualizations...")

    # Load data
    loader = YachtDataLoader('./yacht_hydro.xls')
    df = loader.load()

    preprocessor = DataPreprocessor()
    df_vt = preprocessor.create_vt_surface(df)

    # Figure 1: V-T-R 3D scatter
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(df_vt['V'], df_vt['T'], df_vt['R'],
                        c=df_vt['R'], cmap='viridis', s=30, alpha=0.6)

    ax.set_xlabel('Velocity (knots)', fontsize=12)
    ax.set_ylabel('Draft (meters)', fontsize=12)
    ax.set_zlabel('Resistance', fontsize=12)
    ax.set_title('UCI Yacht Data: Resistance Surface R(V, T)', fontsize=14, fontweight='bold')

    plt.colorbar(scatter, ax=ax, label='Resistance', shrink=0.5)
    plt.tight_layout()
    plt.savefig('../figures/mvp1_yacht_3d_scatter.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/mvp1_yacht_3d_scatter.png")
    plt.close()

    # Figure 2: Train-test split visualization
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(df_vt, test_size=0.2, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(X_train['V'], X_train['T'], c=y_train, cmap='viridis',
              s=50, alpha=0.6, label='Training (246)', marker='o', edgecolors='black')
    ax.scatter(X_test['V'], X_test['T'], c=y_test, cmap='plasma',
              s=100, alpha=0.9, label='Test (62)', marker='s', edgecolors='red', linewidths=2)

    ax.set_xlabel('Velocity (knots)', fontsize=12)
    ax.set_ylabel('Draft (meters)', fontsize=12)
    ax.set_title('Train-Test Split in (V, T) Space (80/20)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/mvp1_train_test_split.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/mvp1_train_test_split.png")
    plt.close()

    # Figure 3: Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    correlation = df_vt.corr()
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm',
                square=True, linewidths=1, cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Correlation Matrix: (V, T, R)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/mvp1_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/mvp1_correlation_matrix.png")
    plt.close()

    print("MVP-1 visualizations complete!\n")


def create_mvp2_visualizations():
    """Generate MVP-2 visualizations."""
    print("Generating MVP-2 visualizations...")

    generator = SyntheticSurfaceGenerator(random_state=42)

    # Figure 1: Synthetic surface (ground truth)
    grid = generator.create_dense_grid(grid_size=100)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(grid['V_grid'], grid['T_grid'], grid['R_grid'],
                           cmap='viridis', alpha=0.8, edgecolor='none')

    ax.set_xlabel('Velocity (knots)', fontsize=12)
    ax.set_ylabel('Draft (meters)', fontsize=12)
    ax.set_zlabel('Resistance', fontsize=12)
    ax.set_title('Synthetic Resistance Surface (Ground Truth)', fontsize=14, fontweight='bold')

    plt.colorbar(surf, ax=ax, label='Resistance', shrink=0.5)
    plt.tight_layout()
    plt.savefig('../figures/mvp2_synthetic_surface.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/mvp2_synthetic_surface.png")
    plt.close()

    # Figure 2: Sampling strategies comparison
    n_samples = 50

    df_random = generator.sample_sparse(n_samples, strategy='random', add_noise=False)
    df_lhs = generator.sample_sparse(n_samples, strategy='latin_hypercube', add_noise=False)
    df_struct = generator.sample_sparse(49, strategy='structured', add_noise=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Random
    axes[0].scatter(df_random['V'], df_random['T'], c='blue', s=50, alpha=0.6, edgecolors='black')
    axes[0].set_xlabel('Velocity (knots)')
    axes[0].set_ylabel('Draft (meters)')
    axes[0].set_title(f'Random Sampling (n={n_samples})', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(9.5, 25.5)
    axes[0].set_ylim(5.5, 10.5)

    # Latin Hypercube
    axes[1].scatter(df_lhs['V'], df_lhs['T'], c='green', s=50, alpha=0.6, edgecolors='black')
    axes[1].set_xlabel('Velocity (knots)')
    axes[1].set_ylabel('Draft (meters)')
    axes[1].set_title(f'Latin Hypercube (n={n_samples})', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(9.5, 25.5)
    axes[1].set_ylim(5.5, 10.5)

    # Structured
    axes[2].scatter(df_struct['V'], df_struct['T'], c='red', s=50, alpha=0.6, edgecolors='black')
    axes[2].set_xlabel('Velocity (knots)')
    axes[2].set_ylabel('Draft (meters)')
    axes[2].set_title(f'Structured + Jitter (n=49)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(9.5, 25.5)
    axes[2].set_ylim(5.5, 10.5)

    plt.tight_layout()
    plt.savefig('../figures/mvp2_sampling_strategies.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/mvp2_sampling_strategies.png")
    plt.close()

    # Figure 3: Noise impact visualization
    V = np.linspace(10, 25, 100)
    T = np.full_like(V, 8.0)  # Fixed draft
    R_clean = generator.resistance_function(V, T)

    R_noisy_30db = generator.add_noise(R_clean, snr_db=30)
    R_noisy_20db = generator.add_noise(R_clean, snr_db=20)
    R_noisy_10db = generator.add_noise(R_clean, snr_db=10)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(V, R_clean, 'k-', linewidth=2, label='Clean (Ground Truth)', zorder=5)
    ax.plot(V, R_noisy_30db, 'g-', alpha=0.7, label='SNR = 30 dB (3% noise)')
    ax.plot(V, R_noisy_20db, 'b-', alpha=0.7, label='SNR = 20 dB (10% noise)')
    ax.plot(V, R_noisy_10db, 'r-', alpha=0.7, label='SNR = 10 dB (32% noise)')

    ax.set_xlabel('Velocity (knots)', fontsize=12)
    ax.set_ylabel('Resistance', fontsize=12)
    ax.set_title('Impact of Noise on Resistance Measurements (T = 8m)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/mvp2_noise_impact.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/mvp2_noise_impact.png")
    plt.close()

    # Figure 4: Coverage comparison (binned)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Create 2D histogram for Random
    h_random, xedges, yedges = np.histogram2d(df_random['V'], df_random['T'], bins=10)
    extent_random = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im1 = axes[0].imshow(h_random.T, extent=extent_random, origin='lower',
                         cmap='Blues', aspect='auto', interpolation='nearest')
    axes[0].scatter(df_random['V'], df_random['T'], c='red', s=20, alpha=0.5, edgecolors='black')
    axes[0].set_xlabel('Velocity (knots)')
    axes[0].set_ylabel('Draft (meters)')
    axes[0].set_title('Random Sampling: Bin Coverage', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Samples per bin')

    # Create 2D histogram for LHS
    h_lhs, xedges, yedges = np.histogram2d(df_lhs['V'], df_lhs['T'], bins=10)
    extent_lhs = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im2 = axes[1].imshow(h_lhs.T, extent=extent_lhs, origin='lower',
                         cmap='Greens', aspect='auto', interpolation='nearest')
    axes[1].scatter(df_lhs['V'], df_lhs['T'], c='red', s=20, alpha=0.5, edgecolors='black')
    axes[1].set_xlabel('Velocity (knots)')
    axes[1].set_ylabel('Draft (meters)')
    axes[1].set_title('Latin Hypercube: Bin Coverage', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Samples per bin')

    plt.tight_layout()
    plt.savefig('../figures/mvp2_coverage_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/mvp2_coverage_comparison.png")
    plt.close()

    print("MVP-2 visualizations complete!\n")


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("  Generating Visualizations for MVP-1 and MVP-2")
    print("="*60 + "\n")

    # Create figures directory if it doesn't exist
    import os
    os.makedirs('../figures', exist_ok=True)

    try:
        create_mvp1_visualizations()
        create_mvp2_visualizations()

        print("="*60)
        print("  All visualizations generated successfully!")
        print("  Check the 'figures/' directory")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
