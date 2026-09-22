"""Global planner logic, without ROS: the same A* the 2D simulator uses."""
import numpy as np

from martha_nav.sim2d.planner import INFLATION, PlanningGrid


class PlannerCore:
    """Map + goal + robot pose -> a route and a status."""

    def __init__(self, inflation=INFLATION, replan_distance=1.0, goal_tolerance=0.3):
        self.inflation = inflation
        self.replan_distance = replan_distance
        self.goal_tolerance = goal_tolerance
        self.planning_grid = None
        self.goal = None
        self.path = None
        self._needs_plan = False

    def set_map(self, grid):
        self.planning_grid = PlanningGrid(grid, self.inflation)
        self._needs_plan = self.goal is not None

    def set_goal(self, x, y):
        self.goal = np.array([float(x), float(y)])
        self.path = None
        self._needs_plan = True

    def update(self, x, y):
        """Advance the state machine; returns idle, active, succeeded or failed."""
        if self.planning_grid is None or self.goal is None:
            return 'idle'
        if np.hypot(x - self.goal[0], y - self.goal[1]) <= self.goal_tolerance:
            return 'succeeded'
        if self.path is not None:
            point = self.path.point_at(self.path.project(x, y))
            if np.hypot(x - point[0], y - point[1]) > self.replan_distance:
                self._needs_plan = True
        if self._needs_plan or self.path is None:
            self.path = self.planning_grid.route((x, y), self.goal)
            self._needs_plan = False
            if self.path is None:
                return 'failed'
        return 'active'
