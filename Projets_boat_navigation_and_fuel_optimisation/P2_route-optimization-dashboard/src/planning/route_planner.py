"""
A* pathfinding for ship route planning.

Implements A* algorithm on a weather-weighted grid to find optimal paths
that balance distance and weather conditions.
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
import heapq
import numpy as np

from src.utils.geometry import Point, grid_to_nautical, nautical_to_grid
from src.models.navigation_grid import NavigationEnvironment


@dataclass(order=True)
class AStarNode:
    """Node for A* priority queue.

    Attributes:
        f_score: Total estimated cost (g + h)
        g_score: Cost from start to this node
        position: (row, col) grid position
        parent: Parent node position (for path reconstruction)
    """
    f_score: float
    g_score: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional[Tuple[int, int]] = field(default=None, compare=False)


class RoutePlanner:
    """A* pathfinding on weather-weighted navigation grid."""

    def __init__(self, environment: NavigationEnvironment):
        """Initialize route planner.

        Args:
            environment: NavigationEnvironment with weather and constraints
        """
        self.env = environment

    def plan_route(self,
                   start: Point,
                   goal: Point,
                   use_weather_penalty: bool = True,
                   connectivity: int = 8) -> Optional[List[Point]]:
        """Plan route from start to goal using A*.

        Args:
            start: Start position in nautical miles
            goal: Goal position in nautical miles
            use_weather_penalty: If True, use weather-based edge costs
            connectivity: 4 (cardinal) or 8 (include diagonals)

        Returns:
            List of waypoints from start to goal, or None if no path exists
        """
        # Convert to grid coordinates
        start_grid = nautical_to_grid(start, self.env.cell_size, self.env.origin)
        goal_grid = nautical_to_grid(goal, self.env.cell_size, self.env.origin)

        # Check if start and goal are navigable
        if not self.env.is_navigable_grid(start_grid):
            raise ValueError(f"Start position {start} is not navigable")
        if not self.env.is_navigable_grid(goal_grid):
            raise ValueError(f"Goal position {goal} is not navigable")

        # A* search
        path_grid = self._astar(start_grid, goal_grid, use_weather_penalty, connectivity)

        if path_grid is None:
            return None

        # Convert grid path to nautical mile waypoints
        path_nm = [
            grid_to_nautical(pos, self.env.cell_size, self.env.origin)
            for pos in path_grid
        ]

        return path_nm

    def _astar(self,
               start: Tuple[int, int],
               goal: Tuple[int, int],
               use_weather_penalty: bool,
               connectivity: int) -> Optional[List[Tuple[int, int]]]:
        """A* algorithm implementation.

        Args:
            start: Start grid position (row, col)
            goal: Goal grid position (row, col)
            use_weather_penalty: Use weather-based costs
            connectivity: 4 or 8

        Returns:
            List of grid positions from start to goal, or None
        """
        # Initialize open and closed sets
        open_set: List[AStarNode] = []
        closed_set: Set[Tuple[int, int]] = set()

        # Track best g_score for each position
        g_scores = {start: 0.0}

        # Initialize with start node
        h_start = self._heuristic(start, goal)
        start_node = AStarNode(
            f_score=h_start,
            g_score=0.0,
            position=start,
            parent=None
        )
        heapq.heappush(open_set, start_node)

        # A* main loop
        while open_set:
            # Get node with lowest f_score
            current = heapq.heappop(open_set)

            # Skip if already processed
            if current.position in closed_set:
                continue

            # Goal reached
            if current.position == goal:
                return self._reconstruct_path(current, g_scores)

            # Mark as processed
            closed_set.add(current.position)

            # Explore neighbors
            neighbors = self.env.get_neighbors(current.position, connectivity=connectivity)

            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue

                # Calculate edge cost
                edge_cost = self._edge_cost(
                    current.position,
                    neighbor,
                    use_weather_penalty
                )

                # Calculate tentative g_score
                tentative_g = current.g_score + edge_cost

                # Check if this is a better path
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g

                    # Calculate f_score (g + h)
                    h_score = self._heuristic(neighbor, goal)
                    f_score = tentative_g + h_score

                    # Add to open set
                    neighbor_node = AStarNode(
                        f_score=f_score,
                        g_score=tentative_g,
                        position=neighbor,
                        parent=current.position
                    )
                    heapq.heappush(open_set, neighbor_node)

        # No path found
        return None

    def _reconstruct_path(self,
                          goal_node: AStarNode,
                          g_scores: dict) -> List[Tuple[int, int]]:
        """Reconstruct path from start to goal.

        Args:
            goal_node: Final node reached
            g_scores: Dictionary mapping positions to g_scores (contains parent info)

        Returns:
            List of grid positions from start to goal
        """
        path = [goal_node.position]
        current = goal_node

        # Walk backwards through parents
        # We need to track parents separately since g_scores doesn't contain them
        # Let's use a different approach: store parent info in nodes during search
        # For now, we'll use a simple BFS reconstruction from goal

        # Actually, let's fix this by maintaining parent pointers properly
        # We need to pass parent information through the nodes

        # Simple fix: reconstruct by finding neighbors with correct g_score
        current_pos = goal_node.position
        current_g = goal_node.g_score

        while current_g > 0:
            # Find parent (neighbor with g_score = current_g - edge_cost)
            neighbors = self.env.get_neighbors(current_pos, connectivity=8)

            found_parent = False
            for neighbor in neighbors:
                if neighbor in g_scores:
                    edge_cost = self._edge_cost(neighbor, current_pos, True)
                    expected_g = g_scores[neighbor] + edge_cost

                    if abs(expected_g - current_g) < 0.001:  # Float comparison
                        path.append(neighbor)
                        current_pos = neighbor
                        current_g = g_scores[neighbor]
                        found_parent = True
                        break

            if not found_parent:
                break  # Shouldn't happen, but prevent infinite loop

        path.reverse()
        return path

    def _heuristic(self,
                   pos: Tuple[int, int],
                   goal: Tuple[int, int]) -> float:
        """Heuristic function for A* (Euclidean distance).

        Args:
            pos: Current position (row, col)
            goal: Goal position (row, col)

        Returns:
            Estimated cost to goal (in grid cells * cell_size)
        """
        dr = goal[0] - pos[0]
        dc = goal[1] - pos[1]
        euclidean_dist = np.sqrt(dr**2 + dc**2)

        # Convert to nautical miles
        return euclidean_dist * self.env.cell_size

    def _edge_cost(self,
                   from_pos: Tuple[int, int],
                   to_pos: Tuple[int, int],
                   use_weather_penalty: bool) -> float:
        """Calculate cost of moving from one position to another.

        Args:
            from_pos: Starting position (row, col)
            to_pos: Ending position (row, col)
            use_weather_penalty: Apply weather-based cost multiplier

        Returns:
            Edge cost in nautical miles (or weather-adjusted)
        """
        # Base cost: Euclidean distance
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        distance = np.sqrt(dr**2 + dc**2) * self.env.cell_size

        if not use_weather_penalty:
            return distance

        # Apply weather penalty at destination
        to_point = grid_to_nautical(to_pos, self.env.cell_size, self.env.origin)
        weather_multiplier = self.env.get_weather_penalty(to_point)

        return distance * weather_multiplier

    def smooth_path(self,
                    path: List[Point],
                    max_iterations: int = 10) -> List[Point]:
        """Smooth a path by removing unnecessary waypoints.

        Uses a simple greedy algorithm: try to connect non-adjacent waypoints
        directly, skipping intermediate points if the direct path is navigable.

        Args:
            path: Original path with many waypoints
            max_iterations: Maximum smoothing passes

        Returns:
            Smoothed path with fewer waypoints
        """
        if len(path) <= 2:
            return path

        smoothed = list(path)

        for _ in range(max_iterations):
            improved = False
            i = 0

            while i < len(smoothed) - 2:
                # Try to connect smoothed[i] directly to smoothed[i+2]
                if self._is_line_navigable(smoothed[i], smoothed[i + 2]):
                    # Remove intermediate point
                    smoothed.pop(i + 1)
                    improved = True
                else:
                    i += 1

            if not improved:
                break  # No more improvements possible

        return smoothed

    def _is_line_navigable(self,
                           start: Point,
                           end: Point,
                           num_samples: int = 10) -> bool:
        """Check if a straight line between two points is navigable.

        Args:
            start: Start point
            end: End point
            num_samples: Number of points to check along the line

        Returns:
            True if all sampled points are navigable
        """
        for i in range(num_samples + 1):
            fraction = i / num_samples
            point = Point(
                start.x + fraction * (end.x - start.x),
                start.y + fraction * (end.y - start.y)
            )

            if not self.env.is_navigable(point):
                return False

        return True

    def get_path_length(self, path: List[Point]) -> float:
        """Calculate total length of a path.

        Args:
            path: List of waypoints

        Returns:
            Total path length in nautical miles
        """
        if len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(path)):
            total_length += path[i - 1].distance_to(path[i])

        return total_length
