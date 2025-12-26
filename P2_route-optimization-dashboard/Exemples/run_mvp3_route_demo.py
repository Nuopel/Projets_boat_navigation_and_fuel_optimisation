"""
Run MVP-3 route optimization demo: NSGA-II over route waypoints.

This demo extends MVP-3 by optimizing route geometry (via intermediate
waypoints) under a storm scenario and plotting Pareto trade-offs.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination

from src.models.ship_model import create_default_ship
from src.models.weather_field import create_storm_scenario
from src.models.navigation_grid import NavigationEnvironment, NavigationConstraints
from src.planning.route_evaluator import RouteEvaluator
from src.planning.constraints import ConstraintChecker
from src.utils.geometry import Point


def _sort_waypoints_along_line(start: Point, goal: Point, waypoints: list[Point]) -> list[Point]:
    """Sort waypoints by projection along the start->goal line."""
    dx = goal.x - start.x
    dy = goal.y - start.y
    denom = dx * dx + dy * dy
    if denom == 0:
        return waypoints

    def projection_t(p: Point) -> float:
        return ((p.x - start.x) * dx + (p.y - start.y) * dy) / denom

    return sorted(waypoints, key=projection_t)


def _segment_samples(start: Point, end: Point, num_samples: int = 8) -> list[Point]:
    """Sample points along a segment for navigability checks."""
    samples = []
    for i in range(num_samples + 1):
        t = i / num_samples
        samples.append(Point(
            start.x + t * (end.x - start.x),
            start.y + t * (end.y - start.y),
        ))
    return samples


def _route_is_navigable(env: NavigationEnvironment, route: list[Point]) -> bool:
    """Check that route stays within navigable waters."""
    for i in range(len(route) - 1):
        for sample in _segment_samples(route[i], route[i + 1]):
            if not env.is_navigable(sample):
                return False
    return True


def main() -> None:
    print("=" * 70)
    print("MVP-3 ROUTE DEMO: NSGA-II Optimizes Route Geometry")
    print("=" * 70)

    Path("outputs").mkdir(exist_ok=True)

    # Scenario setup (same storm as MVP-2)
    grid_shape = (50, 50)
    cell_size = 10.0
    storm_center = Point(250, 250)
    storm_radius = 100.0

    weather = create_storm_scenario(grid_shape, cell_size, storm_center, storm_radius)
    constraints = NavigationConstraints(min_storm_distance=50.0, max_wave_height=4.0)
    env = NavigationEnvironment(grid_shape, cell_size, weather=weather, constraints=constraints)

    ship = create_default_ship()
    evaluator = RouteEvaluator(ship, env)
    checker = ConstraintChecker(env, ship)

    start = Point(50, 50)
    goal = Point(450, 450)
    speed = float(os.environ.get("MVP3_ROUTE_SPEED", "12.0"))

    print(f"Speed (constant): {speed:.2f} kn")
    print(f"Storm max wave: {float(np.max(weather.wave_field)):.2f} m")
    print(f"Max allowed wave: {constraints.max_wave_height:.2f} m")

    # Problem definition: optimize 2 intermediate waypoints (x1,y1,x2,y2)
    n_waypoints = int(os.environ.get("MVP3_ROUTE_WP", "2"))
    n_var = n_waypoints * 2
    lower = np.zeros(n_var)
    upper = np.full(n_var, grid_shape[0] * cell_size)

    class RouteProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=3, n_constr=0, xl=lower, xu=upper)

        def _evaluate(self, X, out, *args, **kwargs):
            objs = []
            for row in X:
                waypoints = []
                for i in range(0, len(row), 2):
                    waypoints.append(Point(float(row[i]), float(row[i + 1])))

                waypoints = _sort_waypoints_along_line(start, goal, waypoints)
                route = [start] + waypoints + [goal]

                if not _route_is_navigable(env, route):
                    objs.append([1e9, 1e9, 1e9])
                    continue

                objectives = evaluator.evaluate_route(route, speed)
                feasible, _ = checker.check_route(route, speed, objectives.time_hours)
                if not feasible:
                    objs.append([1e9, 1e9, 1e9])
                    continue

                objs.append([
                    objectives.time_hours,
                    objectives.fuel_liters,
                    objectives.emissions_kg,
                ])

            out["F"] = np.array(objs)

    pop = int(os.environ.get("MVP3_POP", "20"))
    gen = int(os.environ.get("MVP3_GEN", "30"))
    print(f"NSGA-II params: pop={pop}, gen={gen}, waypoints={n_waypoints}")

    problem = RouteProblem()
    algorithm = NSGA2(pop_size=pop)
    termination = get_termination("n_gen", gen)

    start_time = time.time()

    def _progress_callback(algorithm):
        if algorithm.n_gen == 1 or algorithm.n_gen % 5 == 0 or algorithm.n_gen == gen:
            elapsed = time.time() - start_time
            print(f"  Gen {algorithm.n_gen:>3}/{gen} | elapsed {elapsed:>6.1f}s")

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=False,
        callback=_progress_callback,
    )

    pareto_F = result.F
    pareto_X = result.X

    if pareto_F is None or len(pareto_F) == 0:
        print("No feasible solutions found.")
        return

    print(f"Pareto solutions: {len(pareto_F)}")

    # Scatter: time vs fuel
    times = pareto_F[:, 0]
    fuels = pareto_F[:, 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(times, fuels, s=80, c=times, cmap="viridis", edgecolors="black", alpha=0.8)
    plt.xlabel("Voyage Time (hours)")
    plt.ylabel("Fuel Consumption (liters)")
    plt.title("Route Pareto Front (Time vs Fuel)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/mvp3_route_pareto.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ outputs/mvp3_route_pareto.png")

    # Pick 3 representative routes: best time, best fuel, median
    idx_fast = int(np.argmin(times))
    idx_fuel = int(np.argmin(fuels))
    idx_mid = int(np.argsort(times)[len(times) // 2])

    chosen = [
        ("Fastest", pareto_X[idx_fast], "red"),
        ("Fuel Saver", pareto_X[idx_fuel], "green"),
        ("Balanced", pareto_X[idx_mid], "blue"),
    ]

    # Plot routes over storm wave field
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        weather.wave_field,
        cmap="Blues",
        origin="lower",
        extent=[0, grid_shape[1] * cell_size, 0, grid_shape[0] * cell_size],
        alpha=0.7,
    )
    plt.colorbar(im, ax=ax, label="Wave Height (m)")

    for label, row, color in chosen:
        wps = []
        for i in range(0, len(row), 2):
            wps.append(Point(float(row[i]), float(row[i + 1])))
        wps = _sort_waypoints_along_line(start, goal, wps)
        route = [start] + wps + [goal]
        ax.plot([p.x for p in route], [p.y for p in route], color=color, linewidth=2.5, label=label)

    ax.plot(start.x, start.y, "go", markersize=12, label="Start", markeredgecolor="black")
    ax.plot(goal.x, goal.y, "r^", markersize=12, label="Goal", markeredgecolor="black")
    circle = plt.Circle((storm_center.x, storm_center.y), storm_radius,
                        color="red", fill=False, linestyle=":", linewidth=2, label="Storm Radius")
    ax.add_patch(circle)
    ax.set_title("Pareto-Optimal Routes (Storm Scenario)")
    ax.set_xlabel("Easting (nm)")
    ax.set_ylabel("Northing (nm)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/mvp3_route_paths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ outputs/mvp3_route_paths.png")

    print("=" * 70)
    print("✓ MVP-3 route demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
