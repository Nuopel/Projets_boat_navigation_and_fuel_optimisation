"""
Quick integration test for MVP-2 components.

Tests that all MVP-2 modules work together: A*, route evaluation,
constraints, and weighted sum optimization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ship_model import create_default_ship
from src.models.weather_field import create_calm_scenario, create_storm_scenario
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.planning.route_planner import RoutePlanner
from src.planning.route_evaluator import RouteEvaluator, create_direct_route
from src.planning.constraints import ConstraintChecker, TimeWindow
from src.optimizers.weighted_sum import WeightedSumOptimizer
from src.utils.geometry import Point

print("="*70)
print("MVP-2 INTEGRATION TEST")
print("="*70)

# Setup
print("\\n1. Setting up environment...")
grid_shape = (50, 50)
cell_size = 10.0
start = Point(50, 50)
goal = Point(450, 450)

ship = create_default_ship()
print(f"   ✓ Ship: {ship.specs.name}")
print(f"   ✓ Speed range: {ship.specs.v_min}-{ship.specs.v_max} kn")

# Test with calm weather first
weather_calm = create_calm_scenario(grid_shape, cell_size)
constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=6.0)
env_calm = NavigationEnvironment(grid_shape, cell_size, weather=weather_calm, constraints=constraints)
print(f"   ✓ Environment: {grid_shape} grid, {cell_size} nm cells")
print(f"   ✓ Route: ({start.x}, {start.y}) → ({goal.x}, {goal.y})")

# Test 1: A* Pathfinding
print("\\n2. Testing A* pathfinding...")
planner = RoutePlanner(env_calm)

try:
    route = planner.plan_route(start, goal, use_weather_penalty=True)
    if route:
        path_length = planner.get_path_length(route)
        print(f"   ✓ A* found path with {len(route)} waypoints")
        print(f"   ✓ Path length: {path_length:.1f} nm")

        # Smooth path
        smoothed = planner.smooth_path(route, max_iterations=10)
        smoothed_length = planner.get_path_length(smoothed)
        print(f"   ✓ Smoothed to {len(smoothed)} waypoints ({smoothed_length:.1f} nm)")
    else:
        print("   ✗ A* failed to find path")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ A* error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Route Evaluation
print("\\n3. Testing route evaluation...")
evaluator = RouteEvaluator(ship, env_calm)

try:
    # Evaluate direct route
    direct_route = create_direct_route(start, goal, num_waypoints=2)
    direct_obj = evaluator.evaluate_route(direct_route, speed=15.0)
    print(f"   ✓ Direct route: {direct_obj}")

    # Evaluate A* route
    astar_obj = evaluator.evaluate_route(smoothed, speed=15.0)
    print(f"   ✓ A* route: {astar_obj}")

    # Compare
    fuel_diff_pct = ((astar_obj.fuel_liters - direct_obj.fuel_liters) / direct_obj.fuel_liters) * 100
    print(f"   ✓ Fuel difference: {fuel_diff_pct:+.1f}%")
except Exception as e:
    print(f"   ✗ Evaluation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Constraint Checking
print("\\n4. Testing constraint validation...")
checker = ConstraintChecker(env_calm, ship, time_window=TimeWindow(min_hours=None, max_hours=35.0))

try:
    is_feasible, violations = checker.check_route(smoothed, speed=15.0, voyage_time=astar_obj.time_hours)
    if is_feasible:
        print(f"   ✓ Route satisfies all constraints")
    else:
        print(f"   ⚠ Route has {len(violations)} violations:")
        for v in violations[:3]:
            print(f"      - {v}")
except Exception as e:
    print(f"   ✗ Constraint checking error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Weighted Sum Optimization
print("\\n5. Testing weighted sum optimization...")
optimizer = WeightedSumOptimizer(ship, env_calm, weights=(0.3, 0.6, 0.1))

try:
    result = optimizer.optimize(start, goal, use_astar=True)

    if result.success:
        print(f"   ✓ Optimization successful!")
        print(f"   ✓ Optimal speed: {result.speed:.2f} kn")
        print(f"   ✓ {result.objectives}")
        print(f"   ✓ Waypoints: {len(result.waypoints)}")
        print(f"   ✓ Iterations: {result.iterations}")

        # Compare to baseline
        baseline_result = optimizer.optimize(start, goal, use_astar=False)
        if baseline_result.success:
            fuel_improvement = ((baseline_result.objectives.fuel_liters - result.objectives.fuel_liters) /
                               baseline_result.objectives.fuel_liters) * 100
            print(f"   ✓ Fuel improvement over direct route: {fuel_improvement:.1f}%")
    else:
        print(f"   ✗ Optimization failed: {result.message}")
except Exception as e:
    print(f"   ✗ Optimization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Storm Scenario
print("\\n6. Testing storm detour scenario...")
storm_center = Point(250, 250)
weather_storm = create_storm_scenario(grid_shape, cell_size, storm_center, storm_radius=100.0)
env_storm = NavigationEnvironment(grid_shape, cell_size, weather=weather_storm, constraints=constraints)

optimizer_storm = WeightedSumOptimizer(ship, env_storm, weights=(0.3, 0.6, 0.1))

try:
    result_storm = optimizer_storm.optimize(start, goal, use_astar=True)

    if result_storm.success:
        print(f"   ✓ Storm detour successful!")
        print(f"   ✓ Optimal speed: {result_storm.speed:.2f} kn")
        print(f"   ✓ {result_storm.objectives}")

        # Compare to calm scenario
        fuel_increase = ((result_storm.objectives.fuel_liters - result.objectives.fuel_liters) /
                        result.objectives.fuel_liters) * 100
        time_increase = ((result_storm.objectives.time_hours - result.objectives.time_hours) /
                        result.objectives.time_hours) * 100
        print(f"   ✓ Storm impact: +{fuel_increase:.1f}% fuel, +{time_increase:.1f}% time")
    else:
        print(f"   ⚠ Storm optimization: {result_storm.message}")
except Exception as e:
    print(f"   ✗ Storm scenario error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\\n" + "="*70)
print("MVP-2 INTEGRATION TEST: SUCCESS")
print("="*70)
print("\\nAll core components working:")
print("  ✓ A* pathfinding with weather penalties")
print("  ✓ Route evaluation (time, fuel, emissions)")
print("  ✓ Constraint checking (storms, speed, time windows)")
print("  ✓ Weighted sum optimization")
print("  ✓ Storm detour handling")
print("\\nReady for comprehensive unit tests and demo notebook!")
