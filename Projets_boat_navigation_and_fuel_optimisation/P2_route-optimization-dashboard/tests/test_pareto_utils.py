"""
Unit tests for Pareto dominance and NSGA-II utilities.

Tests dominance checking, non-dominated sorting, crowding distance,
and Pareto front extraction.
"""

import pytest
import numpy as np
from src.optimizers.pareto_utils import (
    ParetoSolution, dominates, fast_non_dominated_sort,
    calculate_crowding_distance, crowding_comparison,
    extract_pareto_front, pareto_front_metrics
)
from src.optimizers.base_optimizer import OptimizationResult
from src.planning.route_evaluator import RouteObjectives
from src.utils.geometry import Point


class TestDominance:
    """Test Pareto dominance checking."""

    def test_strict_dominance(self):
        """Test strict dominance (better in all objectives)."""
        obj1 = (10, 20, 30)
        obj2 = (15, 25, 35)

        assert dominates(obj1, obj2) is True
        assert dominates(obj2, obj1) is False

    def test_partial_dominance(self):
        """Test dominance (better in some, equal in others)."""
        obj1 = (10, 20, 30)
        obj2 = (10, 25, 35)

        assert dominates(obj1, obj2) is True
        assert dominates(obj2, obj1) is False

    def test_equal_solutions(self):
        """Test equal solutions (no dominance)."""
        obj1 = (10, 20, 30)
        obj2 = (10, 20, 30)

        assert dominates(obj1, obj2) is False
        assert dominates(obj2, obj1) is False

    def test_non_dominated_solutions(self):
        """Test non-dominated solutions (trade-off)."""
        obj1 = (10, 25, 30)  # Better time, worse fuel
        obj2 = (15, 20, 35)  # Worse time, better fuel

        assert dominates(obj1, obj2) is False
        assert dominates(obj2, obj1) is False

    def test_two_objectives(self):
        """Test dominance with 2 objectives."""
        obj1 = (10, 20)
        obj2 = (15, 25)

        assert dominates(obj1, obj2) is True

    def test_four_objectives(self):
        """Test dominance with 4 objectives."""
        obj1 = (10, 20, 30, 40)
        obj2 = (15, 25, 35, 45)

        assert dominates(obj1, obj2) is True

    def test_mismatched_dimensions(self):
        """Test error for mismatched dimensions."""
        obj1 = (10, 20)
        obj2 = (15, 25, 35)

        with pytest.raises(ValueError, match="same length"):
            dominates(obj1, obj2)

    def test_edge_case_one_better(self):
        """Test obj1 better in only one objective."""
        obj1 = (10, 25, 35)
        obj2 = (15, 25, 35)

        assert dominates(obj1, obj2) is True


class TestNonDominatedSort:
    """Test non-dominated sorting algorithm."""

    def create_solution(self, objectives: tuple) -> ParetoSolution:
        """Helper to create ParetoSolution."""
        result = OptimizationResult(
            success=True,
            message="Test",
            waypoints=[Point(0, 0), Point(100, 100)],
            speed=10.0,
            objectives=RouteObjectives(*objectives, distance_nm=100.0),
            iterations=0
        )
        return ParetoSolution(objectives=objectives, result=result)

    def test_single_solution(self):
        """Test sorting with single solution."""
        sol = self.create_solution((10, 20, 30))
        fronts = fast_non_dominated_sort([sol])

        assert len(fronts) == 1
        assert len(fronts[0]) == 1
        assert fronts[0][0].rank == 0

    def test_two_dominated_solutions(self):
        """Test sorting with one solution dominating another."""
        sol1 = self.create_solution((10, 20, 30))
        sol2 = self.create_solution((15, 25, 35))

        fronts = fast_non_dominated_sort([sol1, sol2])

        assert len(fronts) == 2
        assert len(fronts[0]) == 1  # Pareto front
        assert len(fronts[1]) == 1  # Dominated

        assert fronts[0][0] == sol1
        assert fronts[1][0] == sol2
        assert sol1.rank == 0
        assert sol2.rank == 1

    def test_two_non_dominated_solutions(self):
        """Test sorting with non-dominated solutions."""
        sol1 = self.create_solution((10, 25, 30))  # Better time
        sol2 = self.create_solution((15, 20, 35))  # Better fuel

        fronts = fast_non_dominated_sort([sol1, sol2])

        assert len(fronts) == 1
        assert len(fronts[0]) == 2
        assert sol1.rank == 0
        assert sol2.rank == 0

    def test_three_solutions_two_fronts(self):
        """Test sorting with three solutions forming two fronts."""
        sol1 = self.create_solution((10, 20, 30))  # Dominates all
        sol2 = self.create_solution((15, 25, 35))  # Dominated by sol1
        sol3 = self.create_solution((12, 30, 32))  # Also dominated by sol1

        fronts = fast_non_dominated_sort([sol1, sol2, sol3])

        assert len(fronts) == 2
        assert len(fronts[0]) == 1
        assert len(fronts[1]) == 2
        assert sol1.rank == 0
        assert sol2.rank == 1
        assert sol3.rank == 1

    def test_mixed_fronts(self):
        """Test sorting with multiple Pareto fronts."""
        # Front 0: (10, 25) and (15, 20)
        # Front 1: (12, 30)
        # Front 2: (20, 35)
        sol1 = self.create_solution((10, 25, 30))
        sol2 = self.create_solution((15, 20, 35))
        sol3 = self.create_solution((12, 30, 32))
        sol4 = self.create_solution((20, 35, 40))

        fronts = fast_non_dominated_sort([sol1, sol2, sol3, sol4])

        assert len(fronts) >= 2
        assert sol1.rank == 0 or sol2.rank == 0  # At least one in front 0
        assert sol4.rank > 0  # Dominated

    def test_empty_list(self):
        """Test sorting empty list."""
        fronts = fast_non_dominated_sort([])

        assert len(fronts) == 1
        assert len(fronts[0]) == 0


class TestCrowdingDistance:
    """Test crowding distance calculation."""

    def create_solution(self, objectives: tuple) -> ParetoSolution:
        """Helper to create ParetoSolution."""
        result = OptimizationResult(
            success=True,
            message="Test",
            waypoints=[Point(0, 0), Point(100, 100)],
            speed=10.0,
            objectives=RouteObjectives(*objectives, distance_nm=100.0),
            iterations=0
        )
        return ParetoSolution(objectives=objectives, result=result)

    def test_empty_front(self):
        """Test crowding distance for empty front."""
        calculate_crowding_distance([])
        # Should not raise error

    def test_single_solution(self):
        """Test crowding distance for single solution."""
        sol = self.create_solution((10, 20, 30))
        calculate_crowding_distance([sol])

        assert sol.crowding_distance == float('inf')

    def test_two_solutions(self):
        """Test crowding distance for two solutions."""
        sol1 = self.create_solution((10, 20, 30))
        sol2 = self.create_solution((15, 25, 35))

        calculate_crowding_distance([sol1, sol2])

        assert sol1.crowding_distance == float('inf')
        assert sol2.crowding_distance == float('inf')

    def test_three_solutions(self):
        """Test crowding distance for three solutions."""
        sol1 = self.create_solution((10, 20, 30))  # Boundary
        sol2 = self.create_solution((15, 25, 35))  # Middle
        sol3 = self.create_solution((20, 30, 40))  # Boundary

        calculate_crowding_distance([sol1, sol2, sol3])

        # Boundary solutions get infinite distance
        assert sol1.crowding_distance == float('inf')
        assert sol3.crowding_distance == float('inf')

        # Middle solution gets finite distance
        assert 0 < sol2.crowding_distance < float('inf')

    def test_uniform_spacing(self):
        """Test crowding distance with uniform spacing."""
        solutions = [
            self.create_solution((10, 20, 30)),
            self.create_solution((15, 25, 35)),
            self.create_solution((20, 30, 40)),
            self.create_solution((25, 35, 45))
        ]

        calculate_crowding_distance(solutions)

        # Boundary solutions
        assert solutions[0].crowding_distance == float('inf')
        assert solutions[-1].crowding_distance == float('inf')

        # Interior solutions should have similar distances (uniform spacing)
        assert solutions[1].crowding_distance > 0
        assert solutions[2].crowding_distance > 0

    def test_crowded_solution(self):
        """Test crowding distance with clustered solutions."""
        solutions = [
            self.create_solution((10, 20, 30)),
            self.create_solution((11, 21, 31)),  # Very close to sol1
            self.create_solution((12, 22, 32)),  # Close
            self.create_solution((30, 40, 50))   # Far away
        ]

        calculate_crowding_distance(solutions)

        # Boundary solutions always infinite
        assert solutions[0].crowding_distance == float('inf')
        assert solutions[-1].crowding_distance == float('inf')

        # Clustered solutions should have smaller distances
        assert solutions[1].crowding_distance < solutions[2].crowding_distance


class TestCrowdingComparison:
    """Test crowding comparison for tournament selection."""

    def create_solution(self, objectives: tuple, rank: int = 0, distance: float = 1.0) -> ParetoSolution:
        """Helper to create ParetoSolution with rank and distance."""
        result = OptimizationResult(
            success=True,
            message="Test",
            waypoints=[Point(0, 0), Point(100, 100)],
            speed=10.0,
            objectives=RouteObjectives(*objectives, distance_nm=100.0),
            iterations=0
        )
        sol = ParetoSolution(objectives=objectives, result=result)
        sol.rank = rank
        sol.crowding_distance = distance
        return sol

    def test_prefer_lower_rank(self):
        """Test preference for lower Pareto rank."""
        sol1 = self.create_solution((10, 20, 30), rank=0, distance=1.0)
        sol2 = self.create_solution((15, 25, 35), rank=1, distance=2.0)

        assert crowding_comparison(sol1, sol2) < 0  # sol1 preferred

    def test_prefer_higher_crowding_same_rank(self):
        """Test preference for higher crowding distance at same rank."""
        sol1 = self.create_solution((10, 20, 30), rank=0, distance=2.0)
        sol2 = self.create_solution((15, 25, 35), rank=0, distance=1.0)

        assert crowding_comparison(sol1, sol2) < 0  # sol1 preferred

    def test_equal_solutions(self):
        """Test equal rank and crowding distance."""
        sol1 = self.create_solution((10, 20, 30), rank=0, distance=1.5)
        sol2 = self.create_solution((15, 25, 35), rank=0, distance=1.5)

        assert crowding_comparison(sol1, sol2) == 0


class TestExtractParetoFront:
    """Test Pareto front extraction."""

    def create_solution(self, objectives: tuple, rank: int = 0) -> ParetoSolution:
        """Helper to create ParetoSolution with rank."""
        result = OptimizationResult(
            success=True,
            message="Test",
            waypoints=[Point(0, 0), Point(100, 100)],
            speed=10.0,
            objectives=RouteObjectives(*objectives, distance_nm=100.0),
            iterations=0
        )
        sol = ParetoSolution(objectives=objectives, result=result)
        sol.rank = rank
        return sol

    def test_extract_single_front(self):
        """Test extracting Pareto front from solutions."""
        sol1 = self.create_solution((10, 20, 30), rank=0)
        sol2 = self.create_solution((15, 25, 35), rank=1)
        sol3 = self.create_solution((12, 22, 32), rank=0)

        solutions = [sol1, sol2, sol3]
        front = extract_pareto_front(solutions)

        assert len(front) == 2
        assert sol1 in front
        assert sol3 in front
        assert sol2 not in front

    def test_extract_empty(self):
        """Test extracting from empty list."""
        front = extract_pareto_front([])
        assert len(front) == 0

    def test_all_dominated(self):
        """Test when no solutions in rank 0."""
        sol1 = self.create_solution((10, 20, 30), rank=1)
        sol2 = self.create_solution((15, 25, 35), rank=1)

        front = extract_pareto_front([sol1, sol2])
        assert len(front) == 0


class TestParetoFrontMetrics:
    """Test Pareto front quality metrics."""

    def create_solution(self, objectives: tuple) -> ParetoSolution:
        """Helper to create ParetoSolution."""
        result = OptimizationResult(
            success=True,
            message="Test",
            waypoints=[Point(0, 0), Point(100, 100)],
            speed=10.0,
            objectives=RouteObjectives(*objectives, distance_nm=100.0),
            iterations=0
        )
        return ParetoSolution(objectives=objectives, result=result)

    def test_empty_front(self):
        """Test metrics for empty front."""
        metrics = pareto_front_metrics([])
        assert metrics['size'] == 0

    def test_single_solution(self):
        """Test metrics for single solution."""
        sol = self.create_solution((10, 20, 30))
        metrics = pareto_front_metrics([sol])

        assert metrics['size'] == 1
        assert metrics['time_min'] == 10
        assert metrics['fuel_min'] == 20
        assert metrics['emissions_min'] == 30

    def test_multiple_solutions(self):
        """Test metrics for multiple solutions."""
        solutions = [
            self.create_solution((10, 25, 35)),
            self.create_solution((15, 20, 30)),
            self.create_solution((20, 30, 40))
        ]

        metrics = pareto_front_metrics(solutions)

        assert metrics['size'] == 3
        assert metrics['time_min'] == 10
        assert metrics['time_max'] == 20
        assert metrics['fuel_min'] == 20
        assert metrics['fuel_max'] == 30
        assert 'time_mean' in metrics
        assert 'time_std' in metrics
        assert 'time_range' in metrics
