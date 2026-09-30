"""Global planner logic, without ROS: the same grid planner the 2D simulator uses."""
import numpy as np
from scipy import ndimage

from martha_nav.sim2d.planner import INFLATION, Path, PlanningGrid


class PlannerCore:
    """Map + goal + robot pose -> a route and a status."""

    def __init__(self, inflation=INFLATION, replan_distance=1.0, goal_tolerance=0.3,
                 snap_distance=0.5):
        self.inflation = inflation
        self.replan_distance = replan_distance
        self.goal_tolerance = goal_tolerance
        # The robot (0.205 m half width) gets closer to walls than the inflation, and a goal
        # can be placed there too. Up to this far they route from the nearest free cell.
        # The free cells beyond a wall are over 0.7 m away (half width + wall + inflation),
        # so the snap never crosses one.
        self.snap_distance = snap_distance
        self.planning_grid = None
        self.cancel()

    def set_map(self, grid):
        self.planning_grid = PlanningGrid(grid, self.inflation)
        # For every cell, the distance to the nearest free cell and that cell's index.
        self._gap, self._nearest = ndimage.distance_transform_edt(
            ~self.planning_grid.free, return_indices=True)
        self._needs_plan = self.goal is not None

    def set_goal(self, x, y):
        self.goal = np.array([float(x), float(y)])
        self.path = None
        self.reached = False
        self._needs_plan = True

    def cancel(self):
        self.goal = None
        self.path = None
        self.reached = False
        self._needs_plan = False

    def update(self, x, y):
        """Advance the state machine; returns idle, active, succeeded or failed."""
        if self.planning_grid is None or self.goal is None:
            return 'idle'
        if self.reached or np.hypot(x - self.goal[0], y - self.goal[1]) <= self.goal_tolerance:
            self.reached = True              # done until the next goal, wherever the robot goes
            return 'succeeded'
        if self.path is not None:
            point = self.path.point_at(self.path.project(x, y))
            if np.hypot(x - point[0], y - point[1]) > self.replan_distance:
                self._needs_plan = True
        if self._needs_plan or self.path is None:
            self.path = self._plan(x, y)
            self._needs_plan = False
            if self.path is None:
                return 'failed'
        return 'active'

    def _plan(self, x, y):
        if self.planning_grid.grid.occupied(*self.goal):
            return None
        start, goal = self._usable(x, y), self._usable(*self.goal)
        if start is None or goal is None:
            return None
        route = self.planning_grid.route(start, goal)
        if route is None:
            return None
        # The route joins free cells; it still has to start at the robot and end at the goal.
        points = route.points
        if not np.allclose(points[0], (x, y)):
            points = np.vstack([[x, y], points])
        if not np.allclose(points[-1], self.goal):
            points = np.vstack([points, self.goal])
        return Path(points)

    def _usable(self, x, y):
        """(x, y) if the planner can use it, else the nearest free cell within snap_distance."""
        free = self.planning_grid.free
        row, col = self.planning_grid.grid.to_cell(x, y)
        if not (0 <= row < free.shape[0] and 0 <= col < free.shape[1]):
            return None
        if free[row, col]:
            return x, y
        if self._gap[row, col] * self.planning_grid.grid.resolution > self.snap_distance:
            return None
        return self.planning_grid.grid.cell_center(*self._nearest[:, row, col])
