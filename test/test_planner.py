import numpy as np

from martha_nav.sim2d.geometry import draw_box, empty_grid
from martha_nav.sim2d.planner import Path, PlanningGrid, carrot


def corridor_grid(gap=None):
    """8 x 3 m room; optional wall across x = 4 with a gap of `gap` metres centred at y = 1.5."""
    g = empty_grid(8.0, 3.0)
    if gap is not None:
        lower = 1.5 - gap / 2
        draw_box(g, 4.0, lower / 2, 0.1, lower)
        draw_box(g, 4.0, 3.0 - lower / 2, 0.1, lower)
    return g


def test_path_arc_length_and_point_at():
    p = Path([[0, 0], [3, 0], [3, 4]])
    assert p.length == 7.0
    assert np.allclose(p.point_at(5.0), [3, 2])
    assert np.allclose(p.point_at(99), [3, 4])


def test_project_uses_window_around_hint():
    p = Path([[0, 0], [4, 0], [4, 1], [0, 1]])     # U-turn: legs 1 m apart
    assert abs(p.project(1.0, 0.1) - 1.0) < 1e-9
    # Near the start of the return leg, with a hint on the first leg, stay on the first leg.
    assert p.project(1.0, 0.6, s_hint=1.0) < 2.0


def test_route_found_in_open_room_and_blocked_by_wall():
    open_pg = PlanningGrid(corridor_grid())
    route = open_pg.route((1.0, 1.5), (7.0, 1.5))
    assert route is not None and 6.0 <= route.length <= 6.2
    blocked = PlanningGrid(corridor_grid(gap=0.0))
    assert blocked.route((1.0, 1.5), (7.0, 1.5)) is None


def test_inflation_requires_gap_of_about_0_8_m():
    assert PlanningGrid(corridor_grid(gap=0.9)).route((1.0, 1.5), (7.0, 1.5)) is not None
    assert PlanningGrid(corridor_grid(gap=0.7)).route((1.0, 1.5), (7.0, 1.5)) is None


def test_route_endpoints_are_exact():
    route = PlanningGrid(corridor_grid()).route((1.01, 1.49), (6.97, 1.52))
    assert np.allclose(route.points[0], [1.01, 1.49])
    assert np.allclose(route.points[-1], [6.97, 1.52])


def test_carrot_lookahead_and_end_of_path():
    p = Path([[0, 0], [10, 0]])
    pt, s = carrot(p, 2.0, 1.5)
    assert np.allclose(pt, [3.5, 0]) and s == 3.5
    pt, s = carrot(p, 9.5, 1.5)
    assert np.allclose(pt, [10, 0])


def test_carrot_skips_points_near_scan_hits():
    p = Path([[0, 0], [10, 0]])
    scan = np.array([[3.5, 0.1], [3.8, 0.0]])       # obstacle right on the carrot
    pt, s = carrot(p, 2.0, 1.5, scan_points=scan, clearance=0.4)
    assert s >= 4.2 - 1e-9
    assert np.min(np.hypot(*(scan - pt).T)) >= 0.4


def test_distance_field_goes_round_a_wall():
    from martha_nav.sim2d.planner import DistanceField
    grid = empty_grid(6.0, 6.0)
    draw_box(grid, 3.0, 2.5, 0.2, 5.0, 0.0)        # wall from the floor up to y = 5, gap above
    field = DistanceField(grid, (5.0, 1.0))
    assert abs(field(5.0, 1.5) - 0.5) < 0.1                     # open space: close to straight line
    around = field(1.0, 1.0)
    assert around > 9.0                  # straight line is 4 m; round the wall end it is ~10 m
    assert field(3.0, 2.5) is None                              # inside the wall


def test_route_progress_counts_only_new_records():
    """Pacing back and forth is not progress: only beating the best arc length is."""
    from martha_nav.sim2d.planner import RouteProgress
    progress = RouteProgress(Path([[0.0, 0.0], [10.0, 0.0]]))
    assert progress.update(1.0, 0.0) > 0.99
    assert progress.seconds_without_progress == 0.0
    for x in (0.5, 1.0, 0.5, 1.0):                  # back and forth, never past 1 m
        assert progress.update(x, 0.0) == 0.0
    assert np.isclose(progress.seconds_without_progress, 0.4)      # 4 control steps of 0.1 s
    progress.update(1.5, 0.0)
    assert progress.seconds_without_progress == 0.0


def test_route_progress_keeps_the_stuck_time_across_a_reroute():
    """A replan to the same goal restarts the arc length, not the time without progress."""
    from martha_nav.sim2d.planner import RouteProgress
    progress = RouteProgress(Path([[0.0, 0.0], [10.0, 0.0]]))
    progress.update(2.0, 0.0)
    for _ in range(5):
        progress.update(2.0, 0.0)
    progress.reroute(Path([[2.0, 0.0], [2.0, 5.0], [10.0, 5.0]]))
    assert progress.s == 0.0
    assert progress.update(2.0, 0.0) == 0.0
    assert np.isclose(progress.seconds_without_progress, 0.6)
    assert progress.update(2.0, 1.0) > 0.99
