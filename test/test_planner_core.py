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


def walled_room():
    """8 x 4 m room with a wall along the top edge (y from 3.8 to 4.0)."""
    g = empty_grid(8.0, 4.0)
    draw_box(g, 4.0, 3.9, 8.0, 0.2)
    return g


def test_the_goal_stays_done_after_success():
    """Moving the robot away afterwards (a teleport between episodes) must not resume it."""
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    assert core.update(6.9, 2.05) == 'succeeded'
    assert core.update(1.0, 2.0) == 'succeeded'


def test_cancel_drops_the_goal():
    core = PlannerCore()
    core.set_map(room())
    core.set_goal(7.0, 2.0)
    core.update(1.0, 2.0)
    core.cancel()
    assert core.update(1.0, 2.0) == 'idle'
    assert core.goal is None and core.path is None


def test_plans_from_a_robot_brushing_a_wall():
    """0.3 m from a wall the robot is inside the 0.4 m inflation, but not stuck."""
    core = PlannerCore()
    core.set_map(walled_room())
    core.set_goal(7.0, 1.5)
    assert core.update(1.0, 3.5) == 'active'
    assert np.allclose(core.path.points[0], [1.0, 3.5])
    assert np.allclose(core.path.points[-1], [7.0, 1.5])


def test_plans_to_a_goal_near_a_wall():
    core = PlannerCore()
    core.set_map(walled_room())
    core.set_goal(7.0, 3.5)
    assert core.update(1.0, 1.5) == 'active'
    assert np.allclose(core.path.points[-1], [7.0, 3.5])


def test_a_goal_inside_an_obstacle_fails():
    core = PlannerCore()
    core.set_map(walled_room())
    core.set_goal(4.0, 3.9)
    assert core.update(1.0, 1.5) == 'failed'


def test_no_jump_to_free_space_far_away():
    """Deep in a 0.6 m dead end there is no usable cell within reach: fail, do not teleport."""
    g = empty_grid(8.0, 4.0)
    draw_box(g, 5.0, 1.65, 6.0, 0.1)               # two walls leave a 0.6 m slot from x = 2 to 8
    draw_box(g, 5.0, 2.35, 6.0, 0.1)
    core = PlannerCore()
    core.set_map(g)
    core.set_goal(1.0, 0.5)
    assert core.update(6.0, 2.0) == 'failed'
