"""
Weighted sum scalarization optimizer.

Combines multiple objectives into a single weighted sum and optimizes
using scipy.optimize. Useful for single-objective baseline and exploring
Pareto front by varying weights.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize, Bounds

from src.utils.geometry import Point
from src.models.ship_model import ShipDynamics
from src.models.navigation_grid import NavigationEnvironment
from src.planning.route_planner import RoutePlanner
from src.planning.route_evaluator import RouteEvaluator, RouteObjectives
from src.planning.constraints import ConstraintChecker, TimeWindow
from src.optimizers.base_optimizer import OptimizerInterface, OptimizationResult


class WeightedSumOptimizer(OptimizerInterface):
    """Weighted sum scalarization: minimize w₁·T + w₂·F + w₃·E.

    First finds a feasible route using A*, then optimizes speed to
    minimize the weighted combination of time, fuel, and emissions.
    """

    def __init__(self,
                 ship: ShipDynamics,
                 environment: NavigationEnvironment,
                 weights: Optional[Tuple[float, float, float]] = None):
        """Initialize weighted sum optimizer.

        Args:
            ship: ShipDynamics model
            environment: NavigationEnvironment with weather
            weights: (w_time, w_fuel, w_emissions) tuple (default: (0.2, 0.7, 0.1))
        """
        self.ship = ship
        self.env = environment
        self.weights = weights or (0.2, 0.7, 0.1)

        # Validate weights
        if len(self.weights) != 3:
            raise ValueError("Weights must be tuple of 3 values")
        if any(w < 0 for w in self.weights):
            raise ValueError("Weights must be non-negative")
        if sum(self.weights) == 0:
            raise ValueError("At least one weight must be positive")

        # Create helper components
        self.planner = RoutePlanner(environment)
        self.evaluator = RouteEvaluator(ship, environment)

    def optimize(self,
                 start: Point,
                 goal: Point,
                 time_window: Optional[TimeWindow] = None,
                 initial_speed: Optional[float] = None,
                 use_astar: bool = True,
                 **kwargs) -> OptimizationResult:
        """Optimize route and speed to minimize weighted objectives.

        Args:
            start: Start position
            goal: Goal position
            time_window: Optional time window constraint
            initial_speed: Initial guess for speed (default: midpoint)
            use_astar: If True, use A* for route; if False, use direct route
            **kwargs: Additional parameters (e.g., max_iterations)

        Returns:
            OptimizationResult with optimized route and speed
        """
        # Step 1: Find feasible route using A*
        if use_astar:
            route = self.planner.plan_route(start, goal, use_weather_penalty=True)

            if route is None:
                return OptimizationResult(
                    waypoints=[start, goal],
                    speed=self.ship.specs.v_min,
                    objectives=RouteObjectives(0, 0, 0, 0),
                    success=False,
                    message="A* pathfinding failed: no feasible route found"
                )

            # Smooth path to reduce waypoints
            route = self.planner.smooth_path(route, max_iterations=10)
        else:
            # Direct route
            route = [start, goal]

        # Step 2: Optimize speed for this route
        optimal_speed, objectives, iterations = self._optimize_speed(
            route,
            time_window,
            initial_speed,
            kwargs.get('max_iterations', 100)
        )

        # Step 3: Validate constraints
        checker = ConstraintChecker(self.env, self.ship, time_window)
        is_feasible, violations = checker.check_route(
            route,
            optimal_speed,
            objectives.time_hours
        )

        if not is_feasible:
            violation_msgs = "\\n".join(str(v) for v in violations[:3])  # Show first 3
            message = f"Route violates constraints:\\n{violation_msgs}"
            if len(violations) > 3:
                message += f"\\n... and {len(violations) - 3} more"
        else:
            message = f"Optimization successful with weights {self.weights}"

        return OptimizationResult(
            waypoints=route,
            speed=optimal_speed,
            objectives=objectives,
            success=is_feasible,
            message=message,
            iterations=iterations,
            metadata={
                'weights': self.weights,
                'num_waypoints': len(route),
                'path_length_nm': self.planner.get_path_length(route),
                'violations': violations if not is_feasible else []
            }
        )

    def _optimize_speed(self,
                        route: List[Point],
                        time_window: Optional[TimeWindow],
                        initial_speed: Optional[float],
                        max_iterations: int) -> Tuple[float, RouteObjectives, int]:
        """Optimize speed for a fixed route.

        Args:
            route: Fixed route waypoints
            time_window: Optional time constraint
            initial_speed: Initial speed guess
            max_iterations: Maximum optimizer iterations

        Returns:
            Tuple of (optimal_speed, objectives, iterations)
        """
        # Initial guess
        if initial_speed is None:
            initial_speed = (self.ship.specs.v_min + self.ship.specs.v_max) / 2.0

        # Speed bounds
        bounds = Bounds(
            lb=self.ship.specs.v_min,
            ub=self.ship.specs.v_max
        )

        # Objective function
        def objective_function(speed_array):
            speed = float(speed_array[0])

            try:
                objectives = self.evaluator.evaluate_route(route, speed)
            except ValueError:
                # Invalid speed
                return 1e10

            # Weighted sum
            w_time, w_fuel, w_emissions = self.weights
            weighted_obj = (
                w_time * objectives.time_hours +
                w_fuel * objectives.fuel_liters / 1000.0 +  # Scale fuel to similar magnitude as time
                w_emissions * objectives.emissions_kg / 1000.0  # Scale emissions
            )

            # Add penalty for time window violations
            if time_window is not None:
                if time_window.min_hours is not None and objectives.time_hours < time_window.min_hours:
                    penalty = 1000.0 * (time_window.min_hours - objectives.time_hours)
                    weighted_obj += penalty
                if time_window.max_hours is not None and objectives.time_hours > time_window.max_hours:
                    penalty = 1000.0 * (objectives.time_hours - time_window.max_hours)
                    weighted_obj += penalty

            return weighted_obj

        # Optimize
        result = minimize(
            objective_function,
            x0=[initial_speed],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )

        optimal_speed = float(result.x[0])
        final_objectives = self.evaluator.evaluate_route(route, optimal_speed)

        return (optimal_speed, final_objectives, result.nit)

    def get_name(self) -> str:
        """Get optimizer name."""
        return f"WeightedSum(w=[{self.weights[0]:.2f}, {self.weights[1]:.2f}, {self.weights[2]:.2f}])"

    def get_parameters(self) -> dict:
        """Get current parameters."""
        return {
            'weights': self.weights,
            'w_time': self.weights[0],
            'w_fuel': self.weights[1],
            'w_emissions': self.weights[2]
        }

    def scan_weight_space(self,
                          start: Point,
                          goal: Point,
                          num_samples: int = 10) -> List[OptimizationResult]:
        """Scan different weight combinations to explore trade-offs.

        Useful for approximating the Pareto front by varying weights.

        Args:
            start: Start position
            goal: Goal position
            num_samples: Number of weight combinations to try

        Returns:
            List of OptimizationResults for different weights
        """
        results = []

        # Generate weight combinations (focus on time-fuel trade-off)
        for i in range(num_samples):
            # Vary time weight from 0 to 1
            w_time = i / (num_samples - 1) if num_samples > 1 else 0.5
            w_fuel = 1.0 - w_time
            w_emissions = 0.1  # Small constant weight

            # Normalize
            total = w_time + w_fuel + w_emissions
            weights = (w_time / total, w_fuel / total, w_emissions / total)

            # Temporarily change weights
            original_weights = self.weights
            self.weights = weights

            try:
                result = self.optimize(start, goal)
                results.append(result)
            finally:
                # Restore original weights
                self.weights = original_weights

        return results
