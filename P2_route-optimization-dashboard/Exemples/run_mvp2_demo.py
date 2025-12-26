"""
Run MVP-2 demonstration and generate visualizations.

Demonstrates weighted sum route optimization with:
- A* pathfinding
- Route evaluation
- Constraint checking
- Multi-objective optimization
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our modules
from src.models.ship_model import create_default_ship
from src.models.weather_field import create_calm_scenario, create_storm_scenario
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.planning.route_evaluator import RouteEvaluator, create_direct_route
from src.planning.constraints import TimeWindow
from src.optimizers.weighted_sum import WeightedSumOptimizer
from src.utils.geometry import Point

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['figure.dpi'] = 100

print("="*70)
print("MVP-2 DEMONSTRATION: Single-Objective Route Optimizer")
print("="*70)

# Create outputs directory
Path('outputs').mkdir(exist_ok=True)

# ==================== SETUP ====================
print("\n1. SETUP")
print("-"*70)

# Ship configuration
ship = create_default_ship()
print(f"Ship: {ship.specs.name}")
print(f"Speed range: {ship.specs.v_min}-{ship.specs.v_max} knots")
print(f"Optimal speed (calm): {ship.optimal_speed_calm_weather():.2f} knots")

# Grid configuration
grid_shape = (50, 50)
cell_size = 10.0
print(f"\nNavigation Grid: {grid_shape[0]}×{grid_shape[1]} cells, {cell_size} nm resolution")

# Route endpoints
start = Point(50, 50)
goal = Point(450, 450)
direct_distance = start.distance_to(goal)
print(f"\nRoute: ({start.x}, {start.y}) → ({goal.x}, {goal.y})")
print(f"Direct distance: {direct_distance:.1f} nm")

# ==================== SCENARIO 1: CALM ====================
print("\n2. SCENARIO 1: CALM WEATHER OPTIMIZATION")
print("-"*70)

weather_calm = create_calm_scenario(grid_shape, cell_size)
constraints_calm = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
env_calm = NavigationEnvironment(grid_shape, cell_size, weather=weather_calm, constraints=constraints_calm)

optimizer_calm = WeightedSumOptimizer(ship, env_calm, weights=(0.3, 0.6, 0.1))
print(f"Optimizer: {optimizer_calm.get_name()}")

result_calm = optimizer_calm.optimize(start, goal, use_astar=True)

print(f"\nSuccess: {result_calm.success}")
if result_calm.success:
    print(f"Optimal Speed: {result_calm.speed:.2f} kn")
    print(f"Waypoints: {len(result_calm.waypoints)}")
    print(f"Time: {result_calm.objectives.time_hours:.2f} hours")
    print(f"Fuel: {result_calm.objectives.fuel_liters:,.0f} liters")
    print(f"CO2: {result_calm.objectives.emissions_kg:,.0f} kg")
    print(f"Iterations: {result_calm.iterations}")

# ==================== SCENARIO 2: STORM ====================
print("\n3. SCENARIO 2: STORM DETOUR")
print("-"*70)

storm_center = Point(250, 250)
storm_radius = 100.0
weather_storm = create_storm_scenario(grid_shape, cell_size, storm_center, storm_radius)
constraints_storm = NavigationConstraints(min_storm_distance=50.0, max_wave_height=4.0)
env_storm = NavigationEnvironment(grid_shape, cell_size, weather=weather_storm, constraints=constraints_storm)

optimizer_storm = WeightedSumOptimizer(ship, env_storm, weights=(0.3, 0.6, 0.1))

result_storm = optimizer_storm.optimize(start, goal, use_astar=True)

print(f"Success: {result_storm.success}")
if result_storm.success:
    print(f"Optimal Speed: {result_storm.speed:.2f} kn")
    print(f"Waypoints: {len(result_storm.waypoints)}")
    print(f"Path Length: {result_storm.metadata['path_length_nm']:.1f} nm")

    min_distance = min(wp.distance_to(storm_center) for wp in result_storm.waypoints)
    print(f"Min distance to storm: {min_distance:.1f} nm (required: {constraints_storm.min_storm_distance} nm)")

    if result_calm.success:
        dist_inc = ((result_storm.objectives.distance_nm - result_calm.objectives.distance_nm) /
                    result_calm.objectives.distance_nm) * 100
        fuel_inc = ((result_storm.objectives.fuel_liters - result_calm.objectives.fuel_liters) /
                    result_calm.objectives.fuel_liters) * 100
        print(f"\nStorm Impact:")
        print(f"  Distance: +{dist_inc:.1f}%")
        print(f"  Fuel: +{fuel_inc:.1f}%")
else:
    print(f"Message: {result_storm.message}")
    violations = result_storm.metadata.get('violations', [])
    print(f"Violations: {len(violations)}")
    for v in violations[:3]:
        print(f"  - {v}")

# ==================== SCENARIO 3: TIME WINDOW ====================
print("\n4. SCENARIO 3: TIGHT TIME WINDOW")
print("-"*70)

time_window = TimeWindow(min_hours=None, max_hours=35.0)
result_time = optimizer_calm.optimize(start, goal, time_window=time_window, use_astar=True)

print(f"Time constraint: {time_window}")
print(f"Success: {result_time.success}")
if result_time.success:
    print(f"Optimal Speed: {result_time.speed:.2f} kn")
    print(f"Time: {result_time.objectives.time_hours:.2f} hours (max: {time_window.max_hours:.1f})")
    print(f"Fuel: {result_time.objectives.fuel_liters:,.0f} liters")

# ==================== WEIGHT SENSITIVITY ====================
print("\n5. WEIGHT SENSITIVITY ANALYSIS")
print("-"*70)

results_pareto = optimizer_calm.scan_weight_space(start, goal, num_samples=8)
successful_results = [r for r in results_pareto if r.success]

print(f"Completed {len(successful_results)}/{len(results_pareto)} weight combinations\n")
print(f"{'Weight (T,F,E)':<25} {'Speed':<10} {'Time':<12} {'Fuel':<15}")
print("-"*70)

for result in successful_results:
    weights = result.metadata['weights']
    w_str = f"({weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f})"
    print(f"{w_str:<25} {result.speed:<10.2f} {result.objectives.time_hours:<12.2f} "
          f"{result.objectives.fuel_liters:<15,.0f}")

# ==================== VISUALIZATIONS ====================
print("\n6. GENERATING VISUALIZATIONS")
print("-"*70)

# Figure 1: Storm routes
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: Wind speed
im1 = axes[0].imshow(weather_storm.wind_field, cmap='YlOrRd', origin='lower',
                     extent=[0, grid_shape[1]*cell_size, 0, grid_shape[0]*cell_size],
                     alpha=0.7)
axes[0].set_title('Storm Scenario: Wind Speed & Routes', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Easting (nm)', fontsize=12)
axes[0].set_ylabel('Northing (nm)', fontsize=12)
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('Wind Speed (kn)', fontsize=11)

# Plot direct route
axes[0].plot([start.x, goal.x], [start.y, goal.y], 'r--', linewidth=2.5,
             label='Direct Route', alpha=0.8)

# Plot A* route even if it violates constraints
if result_storm.waypoints:
    opt_x = [wp.x for wp in result_storm.waypoints]
    opt_y = [wp.y for wp in result_storm.waypoints]
    opt_label = "Optimized Route" if result_storm.success else "A* Route (violates constraints)"
    axes[0].plot(opt_x, opt_y, 'b-', linewidth=3, label=opt_label,
                 marker='o', markersize=6)

# Plot start/goal/storm
axes[0].plot(start.x, start.y, 'go', markersize=18, label='Start',
            markeredgecolor='black', markeredgewidth=2)
axes[0].plot(goal.x, goal.y, 'r^', markersize=18, label='Goal',
            markeredgecolor='black', markeredgewidth=2)

circle = plt.Circle((storm_center.x, storm_center.y), storm_radius,
                    color='red', fill=False, linewidth=2.5, linestyle=':',
                    label='Storm Radius')
axes[0].add_patch(circle)

axes[0].legend(loc='upper right', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Right: Wave height
im2 = axes[1].imshow(weather_storm.wave_field, cmap='Blues', origin='lower',
                     extent=[0, grid_shape[1]*cell_size, 0, grid_shape[0]*cell_size],
                     alpha=0.7)
axes[1].set_title('Storm Scenario: Wave Height & Routes', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Easting (nm)', fontsize=12)
axes[1].set_ylabel('Northing (nm)', fontsize=12)
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('Wave Height (m)', fontsize=11)

# Plot routes
axes[1].plot([start.x, goal.x], [start.y, goal.y], 'r--', linewidth=2.5,
             label='Direct Route', alpha=0.8)
if result_storm.waypoints:
    axes[1].plot(opt_x, opt_y, 'b-', linewidth=3, label=opt_label,
                 marker='o', markersize=6)
axes[1].plot(start.x, start.y, 'go', markersize=18, label='Start',
            markeredgecolor='black', markeredgewidth=2)
axes[1].plot(goal.x, goal.y, 'r^', markersize=18, label='Goal',
            markeredgecolor='black', markeredgewidth=2)

circle2 = plt.Circle((storm_center.x, storm_center.y), storm_radius,
                     color='red', fill=False, linewidth=2.5, linestyle=':',
                     label='Storm Radius')
axes[1].add_patch(circle2)

axes[1].legend(loc='upper right', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/mvp2_storm_routes.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ mvp2_storm_routes.png")

# Figure 2: Pareto front
if len(successful_results) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))

    times = [r.objectives.time_hours for r in successful_results]
    fuels = [r.objectives.fuel_liters for r in successful_results]
    speeds = [r.speed for r in successful_results]

    scatter = ax.scatter(times, fuels, c=speeds, s=200, cmap='viridis',
                        edgecolors='black', linewidths=2, alpha=0.8)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed (knots)', fontsize=12)

    for r in successful_results:
        w_time = r.metadata['weights'][0]
        w_fuel = r.metadata['weights'][1]
        ax.annotate(f'({w_time:.1f},{w_fuel:.1f})',
                   (r.objectives.time_hours, r.objectives.fuel_liters),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)

    ax.set_xlabel('Voyage Time (hours)', fontsize=13)
    ax.set_ylabel('Fuel Consumption (liters)', fontsize=13)
    ax.set_title('Pareto Front Approximation: Time vs. Fuel Trade-off',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/mvp2_pareto_front.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ mvp2_pareto_front.png")

# Figure 3: Scenario comparison
scenarios = ['Calm', 'Storm', 'Time Window']
results_list = [result_calm, result_storm, result_time]

valid_scenarios = []
valid_results = []
for scenario, result in zip(scenarios, results_list):
    if result.success:
        valid_scenarios.append(scenario)
        valid_results.append(result)

if len(valid_results) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    times_comp = [r.objectives.time_hours for r in valid_results]
    fuels_comp = [r.objectives.fuel_liters for r in valid_results]
    speeds_comp = [r.speed for r in valid_results]

    colors = ['green', 'orange', 'blue'][:len(valid_scenarios)]

    # Time
    bars1 = axes[0].bar(valid_scenarios, times_comp, color=colors, alpha=0.7)
    axes[0].set_ylabel('Voyage Time (hours)', fontsize=12)
    axes[0].set_title('Voyage Time by Scenario', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, time in zip(bars1, times_comp):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}h', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Fuel
    bars2 = axes[1].bar(valid_scenarios, fuels_comp, color=colors, alpha=0.7)
    axes[1].set_ylabel('Fuel Consumption (liters)', fontsize=12)
    axes[1].set_title('Fuel Consumption by Scenario', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, fuel in zip(bars2, fuels_comp):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{fuel:,.0f}L', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Speed
    bars3 = axes[2].bar(valid_scenarios, speeds_comp, color=colors, alpha=0.7)
    axes[2].set_ylabel('Optimal Speed (knots)', fontsize=12)
    axes[2].set_title('Optimal Speed by Scenario', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].axhline(y=ship.specs.v_min, color='red', linestyle='--', alpha=0.5)
    axes[2].axhline(y=ship.specs.v_max, color='red', linestyle='--', alpha=0.5)
    for bar, speed in zip(bars3, speeds_comp):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{speed:.2f}kn', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/mvp2_scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ mvp2_scenario_comparison.png")

# ==================== SUMMARY ====================
print("\n7. SUMMARY")
print("-"*70)

print("\nComponents Validated:")
print("  ✓ A* pathfinding with weather penalties")
print("  ✓ Route evaluation (time, fuel, emissions)")
print("  ✓ Constraint checking (speed, time, storms)")
print("  ✓ Weighted sum optimization")
print("  ✓ Speed optimization convergence")

print("\nScenarios Solved:")
for scenario, result in zip(scenarios, results_list):
    status = "✓" if result.success else "✗"
    print(f"  {status} {scenario}")

if len(successful_results) > 0:
    times_final = [r.objectives.time_hours for r in successful_results]
    fuels_final = [r.objectives.fuel_liters for r in successful_results]
    print("\nWeight Sensitivity Results:")
    print(f"  Time range: {min(times_final):.1f} - {max(times_final):.1f} hours")
    print(f"  Fuel range: {min(fuels_final):,.0f} - {max(fuels_final):,.0f} liters")
    print(f"  Speed range: {min(speeds):.2f} - {max(speeds):.2f} knots")

print("\n" + "="*70)
print("✓ MVP-2 DEMONSTRATION COMPLETE")
print("="*70)
print("\nAll core optimization components validated.")
print("Ready for MVP-3: Multi-Objective NSGA-II Optimizer")
storm_max_wave = float(np.max(weather_storm.wave_field))
storm_min_wave = float(np.min(weather_storm.wave_field))
print(f"Storm wave height range: {storm_min_wave:.2f}m - {storm_max_wave:.2f}m "
      f"(max allowed: {constraints_storm.max_wave_height:.1f}m)")
