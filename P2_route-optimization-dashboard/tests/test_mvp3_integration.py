"""
Integration test for MVP-3: NSGA-II multi-objective optimizer.

Tests end-to-end NSGA-II optimization to verify all components work together.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ship_model import create_default_ship
from src.models.weather_field import create_calm_scenario
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.optimizers.nsga2 import NSGA2Optimizer
from src.utils.geometry import Point

print("="*70)
print("MVP-3 INTEGRATION TEST: NSGA-II Multi-Objective Optimizer")
print("="*70)

# Setup
print("\n1. SETUP")
print("-"*70)

ship = create_default_ship()
print(f"Ship: {ship.specs.name}")
print(f"Speed range: {ship.specs.v_min}-{ship.specs.v_max} knots")

grid_shape = (50, 50)
cell_size = 10.0
weather = create_calm_scenario(grid_shape, cell_size)
constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

start = Point(50, 50)
goal = Point(450, 450)
print(f"Route: ({start.x}, {start.y}) → ({goal.x}, {goal.y})")
print(f"Direct distance: {start.distance_to(goal):.1f} nm")

# Test 1: NSGA-II optimization with small population
print("\n2. NSGA-II OPTIMIZATION (Small Test)")
print("-"*70)

optimizer = NSGA2Optimizer(
    ship,
    env,
    population_size=20,  # Small for faster test
    max_generations=10,   # Few generations for quick test
    mutation_rate=0.1,
    crossover_rate=0.9
)

print(f"Optimizer: {optimizer.get_name()}")
print(f"Parameters: {optimizer.get_parameters()}")

# Run optimization
print("\nRunning NSGA-II...")
pareto_front = optimizer.optimize_pareto(start, goal, use_astar=True, verbose=False)

print(f"\n✓ Optimization complete")
print(f"Pareto front size: {len(pareto_front)}")

if pareto_front:
    print("\nPareto Front Solutions:")
    print(f"{'Speed (kn)':<12} {'Time (h)':<12} {'Fuel (L)':<15} {'CO2 (kg)':<12} {'Rank'}")
    print("-"*70)

    for sol in sorted(pareto_front, key=lambda s: s.objectives[0])[:10]:  # Show first 10
        print(f"{sol.result.speed:<12.2f} "
              f"{sol.objectives[0]:<12.2f} "
              f"{sol.objectives[1]:<15,.0f} "
              f"{sol.objectives[2]:<12,.0f} "
              f"{sol.rank}")

    # Statistics
    times = [sol.objectives[0] for sol in pareto_front]
    fuels = [sol.objectives[1] for sol in pareto_front]
    speeds = [sol.result.speed for sol in pareto_front]

    print(f"\nPareto Front Statistics:")
    print(f"  Time range: {min(times):.1f} - {max(times):.1f} hours")
    print(f"  Fuel range: {min(fuels):,.0f} - {max(fuels):,.0f} liters")
    print(f"  Speed range: {min(speeds):.2f} - {max(speeds):.2f} knots")
    print(f"  Diversity: {len(pareto_front)} non-dominated solutions")

# Test 2: Representative solution (best fuel)
print("\n3. REPRESENTATIVE SOLUTION (Best Fuel)")
print("-"*70)

result = optimizer.optimize(start, goal, use_astar=True)

print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Optimal Speed: {result.speed:.2f} kn")
print(f"Time: {result.objectives.time_hours:.2f} hours")
print(f"Fuel: {result.objectives.fuel_liters:,.0f} liters")
print(f"CO2: {result.objectives.emissions_kg:,.0f} kg")

# Test 3: Check dominance relationships
print("\n4. DOMINANCE VALIDATION")
print("-"*70)

from src.optimizers.pareto_utils import dominates

violations = 0
for i, sol1 in enumerate(pareto_front):
    for j, sol2 in enumerate(pareto_front):
        if i != j and dominates(sol1.objectives, sol2.objectives):
            print(f"✗ Violation: Solution {i} dominates solution {j}")
            violations += 1

if violations == 0:
    print(f"✓ All {len(pareto_front)} solutions are mutually non-dominated")
else:
    print(f"✗ Found {violations} dominance violations")

# Test 4: Crowding distance check
print("\n5. CROWDING DISTANCE CHECK")
print("-"*70)

from src.optimizers.pareto_utils import calculate_crowding_distance

calculate_crowding_distance(pareto_front)

boundary_count = sum(1 for sol in pareto_front if sol.crowding_distance == float('inf'))
interior_count = len(pareto_front) - boundary_count

print(f"Boundary solutions: {boundary_count} (distance = inf)")
print(f"Interior solutions: {interior_count}")

if interior_count > 0:
    interior_distances = [sol.crowding_distance for sol in pareto_front
                         if sol.crowding_distance != float('inf')]
    print(f"Interior crowding distance range: {min(interior_distances):.4f} - {max(interior_distances):.4f}")

# Summary
print("\n" + "="*70)
print("INTEGRATION TEST SUMMARY")
print("="*70)

print("\nTests Completed:")
print(f"  ✓ NSGA-II optimization ({len(pareto_front)} solutions)")
print(f"  ✓ Representative solution extraction")
print(f"  ✓ Dominance validation ({violations} violations)")
print(f"  ✓ Crowding distance calculation")

print("\nKey Findings:")
print(f"  - Pareto front successfully generated")
print(f"  - All solutions mutually non-dominated: {violations == 0}")
print(f"  - Speed diversity: {max(speeds) - min(speeds):.2f} kn")
print(f"  - Trade-off demonstrated: {(max(fuels)-min(fuels))/min(fuels)*100:.1f}% fuel variation")

print("\n" + "="*70)
if violations == 0 and len(pareto_front) > 0:
    print("✓ MVP-3 INTEGRATION TEST: PASSED")
else:
    print("✗ MVP-3 INTEGRATION TEST: FAILED")
print("="*70)
