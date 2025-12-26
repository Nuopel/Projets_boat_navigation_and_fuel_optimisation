"""Exploratory Data Analysis visualizations for ship fuel efficiency."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Create output directory
OUTPUT_DIR = Path('outputs/eda')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_fuel_consumption_distribution(df: pd.DataFrame) -> None:
    """Plot 1: Fuel consumption distribution with boxplot by ship type.

    Args:
        df: Input DataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram with KDE
    axes[0].hist(df['fuel_consumption'], bins=40, edgecolor='black',
                 alpha=0.7, color='steelblue')
    axes[0].axvline(df['fuel_consumption'].mean(), color='red',
                   linestyle='--', linewidth=2, label=f'Mean: {df["fuel_consumption"].mean():.0f}')
    axes[0].axvline(df['fuel_consumption'].median(), color='green',
                   linestyle='--', linewidth=2, label=f'Median: {df["fuel_consumption"].median():.0f}')
    axes[0].set_xlabel('Fuel Consumption (tonnes)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Fuel Consumption', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Right: Boxplot by ship type
    ship_types = df['ship_type'].value_counts().index
    box_data = [df[df['ship_type'] == st]['fuel_consumption'].values
                for st in ship_types]

    bp = axes[1].boxplot(box_data, labels=ship_types, patch_artist=True,
                         notch=True, showmeans=True)

    # Color boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_xlabel('Ship Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Fuel Consumption (tonnes)', fontsize=12, fontweight='bold')
    axes[1].set_title('Fuel Consumption by Ship Type', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_fuel_distribution.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / '01_fuel_distribution.png'}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot 2: Correlation matrix heatmap for numerical features.

    Args:
        df: Input DataFrame
    """
    # Select numerical columns
    numerical_cols = ['distance', 'fuel_consumption', 'CO2_emissions', 'engine_efficiency']
    corr_matrix = df[numerical_cols].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)

    ax.set_title('Correlation Matrix: Numerical Features', fontsize=14,
                fontweight='bold', pad=20)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_correlation_matrix.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / '02_correlation_matrix.png'}")
    plt.close()


def plot_weather_impact(df: pd.DataFrame) -> None:
    """Plot 3: Weather impact on fuel consumption (violin plot).

    Args:
        df: Input DataFrame
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Order by severity
    weather_order = ['Calm', 'Moderate', 'Stormy']

    # Violin plot
    parts = ax.violinplot(
        [df[df['weather_conditions'] == w]['fuel_consumption'].values
         for w in weather_order],
        positions=[0, 1, 2],
        widths=0.7,
        showmeans=True,
        showmedians=True
    )

    # Color violins
    colors = ['#98D8C8', '#FFD700', '#FF6B6B']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Overlay scatter points
    for i, weather in enumerate(weather_order):
        weather_data = df[df['weather_conditions'] == weather]['fuel_consumption']
        y = weather_data.values
        x = np.random.normal(i, 0.04, size=len(y))  # Jitter
        ax.scatter(x, y, alpha=0.3, s=20, color=colors[i], edgecolors='black', linewidth=0.5)

    # Statistics annotations
    for i, weather in enumerate(weather_order):
        weather_data = df[df['weather_conditions'] == weather]['fuel_consumption']
        mean_val = weather_data.mean()
        median_val = weather_data.median()
        ax.text(i, weather_data.max() * 1.02,
               f'μ={mean_val:.0f}\nmed={median_val:.0f}',
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(weather_order, fontsize=12, fontweight='bold')
    ax.set_ylabel('Fuel Consumption (tonnes)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Weather Conditions', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Weather on Fuel Consumption', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_weather_impact.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / '03_weather_impact.png'}")
    plt.close()


def plot_distance_vs_fuel(df: pd.DataFrame) -> None:
    """Plot 4: Distance vs fuel consumption scatter, colored by ship type.

    Args:
        df: Input DataFrame
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color palette
    ship_types = df['ship_type'].unique()
    colors = {'Oil Service Boat': '#FF6B6B',
             'Tanker Ship': '#4ECDC4',
             'Fishing Trawler': '#45B7D1',
             'Surfer Boat': '#FFA07A'}

    # Scatter plot by ship type
    for ship_type in ship_types:
        mask = df['ship_type'] == ship_type
        ax.scatter(df[mask]['distance'],
                  df[mask]['fuel_consumption'],
                  label=ship_type,
                  alpha=0.6,
                  s=50,
                  color=colors.get(ship_type, '#333333'),
                  edgecolors='black',
                  linewidth=0.5)

    # Add trend line (overall)
    z = np.polyfit(df['distance'], df['fuel_consumption'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['distance'].min(), df['distance'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
           label=f'Trend: y={z[0]:.1f}x+{z[1]:.0f}')

    # Calculate correlation
    corr = df['distance'].corr(df['fuel_consumption'])
    ax.text(0.05, 0.95, f'Correlation: r={corr:.3f}',
           transform=ax.transAxes,
           fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Distance (nautical miles)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fuel Consumption (tonnes)', fontsize=12, fontweight='bold')
    ax.set_title('Fuel Consumption vs Distance by Ship Type', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_distance_vs_fuel.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / '04_distance_vs_fuel.png'}")
    plt.close()


def plot_route_efficiency(df: pd.DataFrame) -> None:
    """Plot 5: Route efficiency comparison (fuel per nautical mile).

    Args:
        df: Input DataFrame
    """
    # Calculate fuel efficiency (tonnes per nm)
    df_with_efficiency = df.copy()
    df_with_efficiency['fuel_per_nm'] = (df_with_efficiency['fuel_consumption'] /
                                         df_with_efficiency['distance'])

    # Group by route
    route_stats = df_with_efficiency.groupby('route_id').agg({
        'fuel_per_nm': ['mean', 'std', 'count']
    }).reset_index()
    route_stats.columns = ['route_id', 'mean_fuel_per_nm', 'std_fuel_per_nm', 'count']
    route_stats = route_stats.sort_values('mean_fuel_per_nm', ascending=False)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(route_stats)),
                  route_stats['mean_fuel_per_nm'],
                  yerr=route_stats['std_fuel_per_nm'],
                  capsize=5,
                  alpha=0.8,
                  color=['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8'],
                  edgecolor='black',
                  linewidth=1.5)

    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, route_stats.itertuples())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.1f}\n(n={row.count})',
               ha='center', va='bottom', fontsize=10,
               fontweight='bold')

    # Customize
    ax.set_xticks(range(len(route_stats)))
    ax.set_xticklabels(route_stats['route_id'], rotation=15, ha='right', fontsize=11)
    ax.set_ylabel('Fuel Consumption (tonnes/nm)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Route', fontsize=12, fontweight='bold')
    ax.set_title('Route Efficiency: Average Fuel Consumption per Nautical Mile',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line for overall mean
    overall_mean = df_with_efficiency['fuel_per_nm'].mean()
    ax.axhline(overall_mean, color='red', linestyle='--', linewidth=2,
              label=f'Overall Mean: {overall_mean:.1f} t/nm')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_route_efficiency.png', bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / '05_route_efficiency.png'}")
    plt.close()


def main():
    """Generate all EDA visualizations."""
    print("=" * 80)
    print("GENERATING EDA VISUALIZATIONS")
    print("=" * 80)

    # Load data
    df = pd.read_csv('data/raw/ship_fuel_efficiency.csv')
    print(f"\nLoaded {len(df)} observations with {len(df.columns)} features\n")

    # Generate plots
    print("Creating visualizations...")
    plot_fuel_consumption_distribution(df)
    plot_correlation_heatmap(df)
    plot_weather_impact(df)
    plot_distance_vs_fuel(df)
    plot_route_efficiency(df)

    print("\n" + "=" * 80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print(f"✓ Saved to: {OUTPUT_DIR}/")
    print("=" * 80)

    # Print key insights
    print("\nKEY INSIGHTS:")
    print(f"  • Mean fuel consumption: {df['fuel_consumption'].mean():.0f} tonnes")
    print(f"  • Distance-fuel correlation: r={df['distance'].corr(df['fuel_consumption']):.3f}")
    print(f"  • Weather impact (Stormy vs Calm): "
          f"{df[df['weather_conditions']=='Stormy']['fuel_consumption'].mean() / df[df['weather_conditions']=='Calm']['fuel_consumption'].mean():.2f}x")
    print(f"  • Most efficient ship type: "
          f"{(df.groupby('ship_type')['fuel_consumption'].mean() / df.groupby('ship_type')['distance'].mean()).idxmin()}")


if __name__ == "__main__":
    main()
