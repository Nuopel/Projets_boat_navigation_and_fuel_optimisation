"""
Base optimizer interface for route optimization.

Defines the abstract interface that all route optimizers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

from src.utils.geometry import Point
from src.planning.route_evaluator import RouteObjectives


@dataclass
class OptimizationResult:
    """Result of route optimization.

    Attributes:
        waypoints: Optimized route waypoints
        speed: Optimal speed (or speed profile)
        objectives: Route objectives (time, fuel, emissions)
        success: Whether optimization succeeded
        message: Status message or error description
        iterations: Number of optimizer iterations
        metadata: Additional optimizer-specific data
    """
    waypoints: List[Point]
    speed: float
    objectives: RouteObjectives
    success: bool
    message: str
    iterations: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"OptimizationResult[{status}]\\n"
            f"  {self.objectives}\\n"
            f"  Speed: {self.speed:.2f} kn\\n"
            f"  Waypoints: {len(self.waypoints)}\\n"
            f"  Iterations: {self.iterations}\\n"
            f"  Message: {self.message}"
        )


class OptimizerInterface(ABC):
    """Abstract base class for route optimizers.

    All optimizers must implement the optimize() method which takes
    a problem specification and returns an OptimizationResult.
    """

    @abstractmethod
    def optimize(self,
                 start: Point,
                 goal: Point,
                 **kwargs) -> OptimizationResult:
        """Optimize route from start to goal.

        Args:
            start: Start position
            goal: Goal position
            **kwargs: Optimizer-specific parameters

        Returns:
            OptimizationResult with optimized route and objectives
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get optimizer name.

        Returns:
            Human-readable optimizer name
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """Get current optimizer parameters.

        Returns:
            Dictionary of parameter names and values
        """
        return {}
