"""
NSGA-II multi-objective optimizer for ship route optimization.

Implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) for
finding Pareto-optimal trade-offs between time, fuel, and emissions.
"""

from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from src.optimizers.base_optimizer import OptimizerInterface, OptimizationResult
from src.optimizers.pareto_utils import (
    ParetoSolution, dominates, fast_non_dominated_sort,
    calculate_crowding_distance, crowding_comparison, extract_pareto_front,
    pareto_front_metrics
)
from src.planning.route_planner import RoutePlanner
from src.planning.route_evaluator import RouteEvaluator
from src.planning.constraints import ConstraintChecker, TimeWindow
from src.models.ship_model import ShipDynamics
from src.models.navigation_grid import NavigationEnvironment
from src.utils.geometry import Point


class NSGA2Optimizer(OptimizerInterface):
    """NSGA-II multi-objective route optimizer.

    Optimizes routes for multiple conflicting objectives simultaneously:
    - Minimize voyage time
    - Minimize fuel consumption
    - Minimize CO2 emissions

    Returns Pareto front of non-dominated solutions instead of single solution.

    Attributes:
        ship: Ship dynamics model
        env: Navigation environment
        population_size: Number of solutions in population
        max_generations: Maximum iterations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
    """

    def __init__(self,
                 ship: ShipDynamics,
                 environment: NavigationEnvironment,
                 population_size: int = 50,
                 max_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9):
        """Initialize NSGA-II optimizer.

        Args:
            ship: Ship dynamics model
            environment: Navigation environment with weather and constraints
            population_size: Population size (must be even)
            max_generations: Maximum number of generations
            mutation_rate: Mutation probability [0, 1]
            crossover_rate: Crossover probability [0, 1]
        """
        self.ship = ship
        self.env = environment
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Initialize components
        self.planner = RoutePlanner(environment)
        self.evaluator = RouteEvaluator(ship, environment)

    def get_name(self) -> str:
        """Get optimizer name."""
        return f"NSGA-II(pop={self.population_size}, gen={self.max_generations})"

    def get_parameters(self) -> dict:
        """Get optimizer parameters."""
        return {
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'algorithm': 'NSGA-II'
        }

    def optimize(self,
                 start: Point,
                 goal: Point,
                 use_astar: bool = True,
                 time_window: Optional[TimeWindow] = None,
                 **kwargs) -> OptimizationResult:
        """Run NSGA-II optimization to find Pareto front.

        Note: Returns single representative solution (best fuel) from Pareto front.
        Use optimize_pareto() to get full Pareto front.

        Args:
            start: Start position
            goal: Goal position
            use_astar: Use A* pathfinding (vs direct route)
            time_window: Optional time window constraint
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with representative solution (best fuel)
        """
        # Run full Pareto optimization
        pareto_front = self.optimize_pareto(start, goal, use_astar, time_window, **kwargs)

        if not pareto_front:
            return OptimizationResult(
                success=False,
                message="NSGA-II failed to find feasible solutions",
                waypoints=[start, goal],
                speed=self.ship.specs.v_min,
                objectives=None,
                iterations=self.max_generations,
                metadata={'pareto_size': 0}
            )

        # Return best fuel solution as representative
        best_fuel = min(pareto_front, key=lambda s: s.objectives[1])  # Index 1 = fuel

        return best_fuel.result

    def optimize_pareto(self,
                       start: Point,
                       goal: Point,
                       use_astar: bool = True,
                       time_window: Optional[TimeWindow] = None,
                       **kwargs) -> List[ParetoSolution]:
        """Run NSGA-II to find complete Pareto front.

        Args:
            start: Start position
            goal: Goal position
            use_astar: Use A* pathfinding
            time_window: Optional time window constraint
            **kwargs: Additional arguments (e.g., verbose)

        Returns:
            List of ParetoSolution objects forming Pareto front
        """
        verbose = kwargs.get('verbose', False)

        # Step 1: Find route
        if use_astar:
            route = self.planner.plan_route(start, goal, use_weather_penalty=True)
            if route is None:
                if verbose:
                    print("A* pathfinding failed")
                return []
            route = self.planner.smooth_path(route)
        else:
            route = [start, goal]

        # Step 2: Initialize population with diverse speeds
        population = self._initialize_population(route, time_window)

        if verbose:
            print(f"Generation 0: Population size = {len(population)}")

        # Step 3: Evolve population
        for generation in range(self.max_generations):
            # Create offspring
            offspring = self._create_offspring(population, route, time_window)

            # Combine parents and offspring
            combined = population + offspring

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined)

            # Calculate crowding distance for each front
            for front in fronts:
                calculate_crowding_distance(front)

            # Select next generation (elitism)
            population = self._select_next_generation(fronts, self.population_size)

            if verbose and (generation + 1) % 10 == 0:
                pareto_size = len(fronts[0]) if fronts else 0
                print(f"Generation {generation + 1}: Pareto front size = {pareto_size}")

        # Final non-dominated sort to get Pareto front
        fronts = fast_non_dominated_sort(population)
        pareto_front = fronts[0] if fronts else []

        if verbose:
            print(f"\nFinal Pareto front: {len(pareto_front)} solutions")

        return pareto_front

    def _initialize_population(self,
                               route: List[Point],
                               time_window: Optional[TimeWindow]) -> List[ParetoSolution]:
        """Initialize population with diverse speeds.

        Args:
            route: Route waypoints
            time_window: Optional time window constraint

        Returns:
            Initial population of ParetoSolution objects
        """
        population = []
        v_min, v_max = self.ship.specs.v_min, self.ship.specs.v_max

        # Generate diverse speeds using Latin Hypercube Sampling
        for i in range(self.population_size):
            # Stratified sampling
            speed = v_min + (v_max - v_min) * (i + np.random.random()) / self.population_size

            # Evaluate solution
            sol = self._evaluate_solution(route, speed, time_window)
            if sol is not None:
                population.append(sol)

        return population

    def _evaluate_solution(self,
                          route: List[Point],
                          speed: float,
                          time_window: Optional[TimeWindow]) -> Optional[ParetoSolution]:
        """Evaluate a solution (route + speed).

        Args:
            route: Route waypoints
            speed: Ship speed
            time_window: Optional time window constraint

        Returns:
            ParetoSolution if feasible, None otherwise
        """
        # Evaluate objectives
        try:
            objectives_obj = self.evaluator.evaluate_route(route, speed)
        except ValueError:
            # Speed out of bounds
            return None

        # Check constraints
        checker = ConstraintChecker(self.env, self.ship, time_window)
        is_feasible, violations = checker.check_route(
            route, speed, objectives_obj.time_hours
        )

        # Create optimization result
        result = OptimizationResult(
            success=is_feasible,
            message="Feasible solution" if is_feasible else f"{len(violations)} violations",
            waypoints=route,
            speed=speed,
            objectives=objectives_obj,
            iterations=0,
            metadata={
                'violations': violations,
                'num_waypoints': len(route),
                'path_length_nm': objectives_obj.distance_nm
            }
        )

        # Only include feasible solutions
        if not is_feasible:
            return None

        # Create Pareto solution
        objectives = (
            objectives_obj.time_hours,
            objectives_obj.fuel_liters,
            objectives_obj.emissions_kg
        )

        return ParetoSolution(objectives=objectives, result=result)

    def _create_offspring(self,
                         population: List[ParetoSolution],
                         route: List[Point],
                         time_window: Optional[TimeWindow]) -> List[ParetoSolution]:
        """Create offspring through selection, crossover, and mutation.

        Args:
            population: Current population
            route: Route waypoints
            time_window: Optional time window

        Returns:
            Offspring population
        """
        offspring = []

        while len(offspring) < self.population_size:
            # Binary tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover
            if np.random.random() < self.crossover_rate:
                child1_speed, child2_speed = self._crossover(
                    parent1.result.speed,
                    parent2.result.speed
                )
            else:
                child1_speed = parent1.result.speed
                child2_speed = parent2.result.speed

            # Mutation
            child1_speed = self._mutate(child1_speed)
            child2_speed = self._mutate(child2_speed)

            # Evaluate children
            child1 = self._evaluate_solution(route, child1_speed, time_window)
            child2 = self._evaluate_solution(route, child2_speed, time_window)

            if child1 is not None:
                offspring.append(child1)
            if child2 is not None and len(offspring) < self.population_size:
                offspring.append(child2)

        return offspring[:self.population_size]

    def _tournament_selection(self, population: List[ParetoSolution]) -> ParetoSolution:
        """Binary tournament selection.

        Randomly select 2 solutions, return better one based on:
        1. Pareto rank (lower is better)
        2. Crowding distance (higher is better)

        Args:
            population: Current population

        Returns:
            Selected solution
        """
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        sol1, sol2 = population[idx1], population[idx2]

        if crowding_comparison(sol1, sol2) <= 0:
            return sol1
        else:
            return sol2

    def _crossover(self, speed1: float, speed2: float) -> Tuple[float, float]:
        """Simulated Binary Crossover (SBX) for real-valued speeds.

        Args:
            speed1: Parent 1 speed
            speed2: Parent 2 speed

        Returns:
            Tuple of (child1_speed, child2_speed)
        """
        eta = 20  # Distribution index

        if abs(speed1 - speed2) < 1e-6:
            return speed1, speed2

        # Random number
        u = np.random.random()

        # SBX formula
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

        child1 = 0.5 * ((1 + beta) * speed1 + (1 - beta) * speed2)
        child2 = 0.5 * ((1 - beta) * speed1 + (1 + beta) * speed2)

        # Clip to bounds
        v_min, v_max = self.ship.specs.v_min, self.ship.specs.v_max
        child1 = np.clip(child1, v_min, v_max)
        child2 = np.clip(child2, v_min, v_max)

        return child1, child2

    def _mutate(self, speed: float) -> float:
        """Polynomial mutation for speed.

        Args:
            speed: Original speed

        Returns:
            Mutated speed
        """
        if np.random.random() > self.mutation_rate:
            return speed

        v_min, v_max = self.ship.specs.v_min, self.ship.specs.v_max
        eta = 20  # Distribution index

        # Normalized speed
        delta = (speed - v_min) / (v_max - v_min)

        # Random number
        u = np.random.random()

        # Polynomial mutation
        if u < 0.5:
            delta_q = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

        # Apply mutation
        speed_mutated = speed + delta_q * (v_max - v_min) * 0.1  # 10% range

        # Clip to bounds
        return np.clip(speed_mutated, v_min, v_max)

    def _select_next_generation(self,
                                fronts: List[List[ParetoSolution]],
                                target_size: int) -> List[ParetoSolution]:
        """Select next generation using elitism.

        Include entire fronts until target size reached.
        For last front, sort by crowding distance.

        Args:
            fronts: List of Pareto fronts (sorted by rank)
            target_size: Desired population size

        Returns:
            Next generation population
        """
        next_gen = []

        for front in fronts:
            if len(next_gen) + len(front) <= target_size:
                # Include entire front
                next_gen.extend(front)
            else:
                # Include partial front sorted by crowding distance
                remaining = target_size - len(next_gen)
                sorted_front = sorted(front, key=lambda s: s.crowding_distance, reverse=True)
                next_gen.extend(sorted_front[:remaining])
                break

        return next_gen

    def compare_with_weighted_sum(self,
                                  start: Point,
                                  goal: Point,
                                  weight_samples: int = 10) -> dict:
        """Compare NSGA-II Pareto front with weighted sum approach.

        Args:
            start: Start position
            goal: Goal position
            weight_samples: Number of weighted sum samples

        Returns:
            Dictionary with comparison metrics
        """
        from src.optimizers.weighted_sum import WeightedSumOptimizer

        # Get NSGA-II Pareto front
        print("Running NSGA-II...")
        pareto_nsga2 = self.optimize_pareto(start, goal, use_astar=True, verbose=True)

        # Get weighted sum Pareto approximation
        print(f"\nRunning Weighted Sum ({weight_samples} samples)...")
        ws_optimizer = WeightedSumOptimizer(self.ship, self.env)
        pareto_ws_results = ws_optimizer.scan_weight_space(start, goal, num_samples=weight_samples)
        pareto_ws = [
            ParetoSolution(
                objectives=(r.objectives.time_hours, r.objectives.fuel_liters, r.objectives.emissions_kg),
                result=r
            )
            for r in pareto_ws_results if r.success
        ]

        # Compute metrics
        metrics_nsga2 = pareto_front_metrics(pareto_nsga2)
        metrics_ws = pareto_front_metrics(pareto_ws)

        return {
            'nsga2': {
                'front': pareto_nsga2,
                'metrics': metrics_nsga2
            },
            'weighted_sum': {
                'front': pareto_ws,
                'metrics': metrics_ws
            },
            'comparison': {
                'size_ratio': metrics_nsga2['size'] / max(metrics_ws['size'], 1),
                'nsga2_size': metrics_nsga2['size'],
                'ws_size': metrics_ws['size']
            }
        }
