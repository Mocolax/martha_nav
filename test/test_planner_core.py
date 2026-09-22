import numpy as np

from martha_nav.ros.planner_core import PlannerCore
from martha_nav.sim2d.geometry import draw_box, empty_grid


def room(blocked=False):
    g = empty_grid(8.0, 4.0)
    if blocked:
        draw_box(g, 4.0, 2.0, 0.2, 4.0)
    return g


def test_idle_without_map_or_goal():
    core = PlannerCore()
    assert core.update(1.0, 2.0) == 'idle'
    core.set_map(room())
    assert core.update(1.0, 2.0) == 'idle'


def test_plans_once_the_goal_is_set():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    assert core.update(1.0, 2.0) == 'active'
    assert core.path is not None and core.path.length > 5.5


def test_reports_failure_when_there_is_no_route():
    core = PlannerCore()
    core.set_map(room(blocked=True))
    core.set_goal(7.0, 2.0)
    assert core.update(1.0, 2.0) == 'failed'
    assert core.path is None


def test_succeeds_inside_the_tolerance():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    assert core.update(6.9, 2.05) == 'succeeded'


def test_replans_only_when_far_from_the_route():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    first = core.path
    core.update(2.0, 2.05)
    assert core.path is first                      # still on the route
    core.update(2.0, 3.4)
    assert core.path is not first                  # 1.4 m away: replanned


def test_a_new_goal_replans():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    first = core.path
    core.set_goal(7.0, 3.0)
    core.update(1.0, 2.0)
    assert core.path is not first
    assert np.allclose(core.path.points[-1], [7.0, 3.0])
