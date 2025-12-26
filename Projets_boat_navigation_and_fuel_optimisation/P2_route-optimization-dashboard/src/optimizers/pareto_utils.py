"""
Pareto dominance and multi-objective optimization utilities.

Provides functions for:
- Pareto dominance checking
- Non-dominated sorting
- Crowding distance calculation
- Pareto front extraction
"""

from typing import List, Tuple, Set
import numpy as np
from dataclasses import dataclass

from src.optimizers.base_optimizer import OptimizationResult


@dataclass
class ParetoSolution:
    """Solution in multi-objective space.

    Attributes:
        objectives: Tuple of objective values (time, fuel, emissions)
        result: Full optimization result
        rank: Pareto rank (0 = non-dominated)
        crowding_distance: Crowding distance for diversity
    """
    objectives: Tuple[float, float, float]
    result: OptimizationResult
    rank: int = 0
    crowding_distance: float = 0.0

    def __repr__(self) -> str:
        return f"ParetoSolution(rank={self.rank}, objectives={self.objectives})"


def dominates(obj1: Tuple[float, ...], obj2: Tuple[float, ...]) -> bool:
    """Check if obj1 Pareto-dominates obj2.

    obj1 dominates obj2 if:
    - obj1 is no worse than obj2 in all objectives
    - obj1 is strictly better than obj2 in at least one objective

    For minimization problems (all objectives).

    Args:
        obj1: First objective vector
        obj2: Second objective vector

    Returns:
        True if obj1 dominates obj2

    Example:
        >>> dominates((10, 20, 30), (15, 25, 35))  # obj1 better in all
        True
        >>> dominates((10, 20, 30), (10, 20, 30))  # Equal
        False
        >>> dominates((10, 25, 30), (15, 20, 35))  # Trade-off
        False
    """
    if len(obj1) != len(obj2):
        raise ValueError(f"Objective vectors must have same length: {len(obj1)} vs {len(obj2)}")

    # Check if obj1 is no worse in all objectives
    no_worse = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))

    # Check if obj1 is strictly better in at least one objective
    strictly_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))

    return no_worse and strictly_better


def fast_non_dominated_sort(solutions: List[ParetoSolution]) -> List[List[ParetoSolution]]:
    """Fast non-dominated sorting algorithm from NSGA-II.

    Assigns Pareto ranks to solutions:
    - Rank 0: Non-dominated solutions (Pareto front)
    - Rank 1: Dominated only by rank 0
    - Rank k: Dominated only by ranks 0..k-1

    Args:
        solutions: List of solutions to sort

    Returns:
        List of fronts, where front[0] is Pareto front (rank 0)

    Complexity: O(MN²) where M = #objectives, N = #solutions
    """
    n = len(solutions)

    # Domination tracking
    domination_count = [0] * n  # How many solutions dominate i
    dominated_solutions = [set() for _ in range(n)]  # Solutions dominated by i

    fronts: List[List[ParetoSolution]] = [[]]

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            obj_i = solutions[i].objectives
            obj_j = solutions[j].objectives

            if dominates(obj_i, obj_j):
                # i dominates j
                dominated_solutions[i].add(j)
                domination_count[j] += 1
            elif dominates(obj_j, obj_i):
                # j dominates i
                dominated_solutions[j].add(i)
                domination_count[i] += 1

    # First front: solutions with domination_count = 0
    for i in range(n):
        if domination_count[i] == 0:
            solutions[i].rank = 0
            fronts[0].append(solutions[i])

    # Subsequent fronts
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []

        for solution in fronts[current_front]:
            solution_idx = solutions.index(solution)

            # For each solution dominated by current solution
            for dominated_idx in dominated_solutions[solution_idx]:
                domination_count[dominated_idx] -= 1

                # If no longer dominated, add to next front
                if domination_count[dominated_idx] == 0:
                    solutions[dominated_idx].rank = current_front + 1
                    next_front.append(solutions[dominated_idx])

        if next_front:
            fronts.append(next_front)
        current_front += 1

    return fronts


def calculate_crowding_distance(front: List[ParetoSolution]) -> None:
    """Calculate crowding distance for solutions in a front.

    Crowding distance measures density of solutions around a point.
    Higher values = more isolated = preserve diversity.

    Boundary solutions get infinite distance (always preserved).

    Modifies solutions in-place by setting crowding_distance attribute.

    Args:
        front: List of solutions in same Pareto rank

    Complexity: O(M*N*log(N)) where M = #objectives, N = #solutions
    """
    n = len(front)

    if n == 0:
        return

    if n <= 2:
        # Boundary cases
        for solution in front:
            solution.crowding_distance = float('inf')
        return

    # Initialize distances to 0
    for solution in front:
        solution.crowding_distance = 0.0

    # Get number of objectives
    n_objectives = len(front[0].objectives)

    # Calculate crowding distance for each objective
    for obj_idx in range(n_objectives):
        # Sort by this objective
        sorted_front = sorted(front, key=lambda s: s.objectives[obj_idx])

        # Boundary solutions get infinite distance
        sorted_front[0].crowding_distance = float('inf')
        sorted_front[-1].crowding_distance = float('inf')

        # Get objective range
        obj_min = sorted_front[0].objectives[obj_idx]
        obj_max = sorted_front[-1].objectives[obj_idx]
        obj_range = obj_max - obj_min

        # Avoid division by zero
        if obj_range == 0:
            continue

        # Calculate crowding distance for interior points
        for i in range(1, n - 1):
            if sorted_front[i].crowding_distance != float('inf'):
                # Distance = sum of normalized distances to neighbors
                distance = (sorted_front[i + 1].objectives[obj_idx] -
                           sorted_front[i - 1].objectives[obj_idx]) / obj_range
                sorted_front[i].crowding_distance += distance


def crowding_comparison(sol1: ParetoSolution, sol2: ParetoSolution) -> int:
    """Compare two solutions for crowding tournament selection.

    Prefers:
    1. Lower rank (better Pareto front)
    2. Higher crowding distance (more diverse)

    Args:
        sol1: First solution
        sol2: Second solution

    Returns:
        -1 if sol1 is preferred, 1 if sol2 is preferred, 0 if equal
    """
    if sol1.rank < sol2.rank:
        return -1
    elif sol1.rank > sol2.rank:
        return 1
    else:
        # Same rank, compare crowding distance
        if sol1.crowding_distance > sol2.crowding_distance:
            return -1
        elif sol1.crowding_distance < sol2.crowding_distance:
            return 1
        else:
            return 0


def extract_pareto_front(solutions: List[ParetoSolution]) -> List[ParetoSolution]:
    """Extract Pareto front (rank 0) from solutions.

    Args:
        solutions: List of solutions with assigned ranks

    Returns:
        List of non-dominated solutions (rank 0)
    """
    return [sol for sol in solutions if sol.rank == 0]


def compute_hypervolume(front: List[ParetoSolution],
                       reference_point: Tuple[float, float, float]) -> float:
    """Compute hypervolume indicator for Pareto front quality.

    Hypervolume = volume of objective space dominated by front.
    Higher is better (covers more objective space).

    Uses simple Monte Carlo approximation for 3D case.

    Args:
        front: Pareto front solutions
        reference_point: Worst point for normalization (time, fuel, emissions)

    Returns:
        Approximate hypervolume (0-1, normalized)
    """
    if not front:
        return 0.0

    if len(front) == 1:
        # Single solution: box from solution to reference
        obj = front[0].objectives
        volume = 1.0
        for i in range(len(obj)):
            volume *= max(0, (reference_point[i] - obj[i]) / reference_point[i])
        return volume

    # Monte Carlo sampling for 3+ objectives
    n_samples = 10000
    n_dominated = 0

    # Get bounds
    obj_min = [min(sol.objectives[i] for sol in front) for i in range(3)]

    for _ in range(n_samples):
        # Random point in objective space
        point = tuple(
            obj_min[i] + np.random.random() * (reference_point[i] - obj_min[i])
            for i in range(3)
        )

        # Check if any solution dominates this point
        for sol in front:
            if all(sol.objectives[i] <= point[i] for i in range(3)):
                n_dominated += 1
                break

    # Normalize by volume
    total_volume = 1.0
    for i in range(3):
        total_volume *= (reference_point[i] - obj_min[i]) / reference_point[i]

    return (n_dominated / n_samples) * total_volume


def pareto_front_metrics(front: List[ParetoSolution]) -> dict:
    """Compute quality metrics for Pareto front.

    Args:
        front: Pareto front solutions

    Returns:
        Dictionary with metrics:
        - size: Number of solutions
        - objectives_min/max/mean: Statistics per objective
        - spread: Range of solutions
    """
    if not front:
        return {'size': 0}

    n_obj = len(front[0].objectives)
    objectives = [[sol.objectives[i] for sol in front] for i in range(n_obj)]

    obj_names = ['time', 'fuel', 'emissions']

    metrics = {
        'size': len(front),
    }

    for i, name in enumerate(obj_names[:n_obj]):
        metrics[f'{name}_min'] = min(objectives[i])
        metrics[f'{name}_max'] = max(objectives[i])
        metrics[f'{name}_mean'] = np.mean(objectives[i])
        metrics[f'{name}_std'] = np.std(objectives[i])
        metrics[f'{name}_range'] = max(objectives[i]) - min(objectives[i])

    return metrics
