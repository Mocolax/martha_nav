import numpy as np
import pytest

from martha_nav.sim2d.planner import PlanningGrid
from martha_nav.sim2d.scenarios import (EVAL_ONLY_SOURCES, TEMPLATES, TRAIN_SOURCES,
                                        ScenarioConfig, generate, point_pairs)


def test_lab_is_never_a_training_source():
    assert 'lab' in EVAL_ONLY_SOURCES
    assert 'lab' not in TRAIN_SOURCES
    assert 'open_room' not in TRAIN_SOURCES


def test_same_seed_same_episode():
    a, b = generate(123), generate(123)
    assert a.source == b.source
    assert np.allclose(a.start, b.start) and np.allclose(a.goal, b.goal)
    assert np.array_equal(a.full.occ, b.full.occ)


@pytest.mark.parametrize('name', list(TEMPLATES))
def test_every_template_yields_episodes(name):
    for seed in range(5):
        sc = generate(seed, ScenarioConfig(sources=(name,), obstacle_mode='none'))
        assert 3.0 <= sc.path.length <= 12.5


def test_obstacle_modes():
    none = [generate(s, ScenarioConfig(obstacle_mode='none')) for s in range(20)]
    assert all(len(sc.obstacles) == 0 for sc in none)
    always = [generate(s, ScenarioConfig(obstacle_mode='always')) for s in range(20)]
    assert sum(len(sc.obstacles) > 0 for sc in always) >= 18


def test_500_mixed_episodes_respect_every_rule():
    cfg = ScenarioConfig()
    with_obstacles = 0
    for seed in range(500):
        sc = generate(seed, cfg)
        assert sc.source in TRAIN_SOURCES
        assert cfg.route_min <= sc.path.length <= cfg.route_max + 0.5
        for ob in sc.obstacles:
            assert np.hypot(ob.x - sc.start[0], ob.y - sc.start[1]) - ob.extent >= cfg.start_clearance
            assert np.hypot(ob.x - sc.goal[0], ob.y - sc.goal[1]) - ob.extent >= cfg.goal_clearance
        if sc.obstacles:
            with_obstacles += 1
            detour = PlanningGrid(sc.full, cfg.inflation).route(sc.start[:2], sc.goal)
            assert detour is not None                        # gap >= 0.8 m exists
            assert detour.length <= 1.5 * sc.path.length + 2.0
    assert 0.6 <= with_obstacles / 500 <= 0.9               # ~80% nominal minus drops


def test_point_pairs_of_the_lab_are_reachable():
    pairs = point_pairs('lab')
    assert 40 <= len(pairs) <= 90                      # 10 points, minus unreachable/short pairs
    for start, goal in pairs[:5]:
        assert len(start) == 2 and len(goal) == 2


def test_a_seed_maps_to_one_fixed_pair():
    cfg = ScenarioConfig(sources=('lab',), point_pairs=point_pairs('lab'), obstacle_mode='always')
    pairs = cfg.point_pairs
    for seed in (1_000_000, 1_000_001, 1_000_000 + len(pairs)):
        sc = generate(seed, cfg)
        start, goal = pairs[seed % len(pairs)]
        assert np.allclose(sc.start[:2], start) and np.allclose(sc.goal, goal)
        assert sc.source == 'lab'
    # Same seed, same obstacles: the episode is reproducible in 2D and in Gazebo.
    a, b = generate(1_000_005, cfg), generate(1_000_005, cfg)
    assert [(o.x, o.y) for o in a.obstacles] == [(o.x, o.y) for o in b.obstacles]


def test_point_episodes_have_a_route():
    cfg = ScenarioConfig(sources=('lab',), point_pairs=point_pairs('lab'), obstacle_mode='none')
    for seed in range(1_000_000, 1_000_010):
        sc = generate(seed, cfg)
        assert sc.path.length >= 1.0
