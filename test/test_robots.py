import json
from dataclasses import asdict

import numpy as np
import pytest
import yaml

from martha_nav.robots import ROBOTS, check_profile, checkpoint_robot
from martha_nav.ros.policy_core import footprint_blocked


def test_martha_keeps_the_numbers_the_code_used_before_the_profiles():
    m = ROBOTS['martha']
    assert (m.length, m.width, m.footprint_offset_x, m.lidar_offset_x) == (0.56, 0.41, 0.0, 0.2325)
    assert (m.lidar_range, m.lidar_min, m.lidar_rate) == (8.0, 0.0, 10.0)
    assert (m.v_max, m.v_reverse, m.v_lateral, m.w_max, m.inflation) == (0.35, 0.15, 0.25, 0.8, 0.40)
    assert m.guard_margin == 0.05
    assert m.holonomic


def test_a_checkpoint_says_which_robot_it_drives(tmp_path):
    model = tmp_path / 'best_model.zip'
    assert checkpoint_robot(model) == 'martha'                          # no config.yaml
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'n_rays': 180}}))
    assert checkpoint_robot(model) == 'martha'                          # trained before profiles
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'martha_tower'}}))
    assert checkpoint_robot(model) == 'martha_tower'
    exported = tmp_path / 'policy.npz'
    np.savez(exported, settings=json.dumps({'robot': 'martha_tower'}))
    assert checkpoint_robot(exported) == 'martha_tower'


def test_a_recorded_profile_must_be_the_current_one():
    martha_tower = asdict(ROBOTS['martha_tower'])
    check_profile(martha_tower, 'martha_tower', 'best_model.zip')
    check_profile(None, 'martha_tower', 'best_model.zip')                     # trained before profiles
    changed = {**martha_tower, 'lidar_min': 0.16, 'guard_margin': 0.15}
    with pytest.raises(ValueError, match=r'different martha_tower profile \(changed: lidar_min, '
                                         r'guard_margin\): retrain and re-export'):
        check_profile(changed, 'martha_tower', 'best_model.zip')
    with pytest.raises(ValueError, match='best_model.zip'):
        check_profile({k: v for k, v in martha_tower.items() if k != 'lidar_min'}, 'martha_tower',
                      'best_model.zip')


def test_the_inflation_is_the_planners_not_the_models():
    """A model trained with other routes still drives on the current planner's."""
    martha_tower = asdict(ROBOTS['martha_tower'])
    check_profile({**martha_tower, 'inflation': 0.30}, 'martha_tower', 'best_model.zip')
    check_profile({k: v for k, v in martha_tower.items() if k != 'inflation'}, 'martha_tower', 'best_model.zip')


@pytest.mark.parametrize('name', list(ROBOTS))
def test_the_guard_reaches_beyond_the_lidars_blind_zone(name):
    """An obstacle that has just left the blind zone must already be inside the guard."""
    r = ROBOTS[name]
    reach = r.length / 2 + r.footprint_offset_x - r.lidar_offset_x + r.guard_margin
    assert 'front' in footprint_blocked([reach - 1e-6], [0.0], r)
    assert 'front' not in footprint_blocked([reach + 1e-3], [0.0], r)
    assert reach - r.lidar_min >= 0.04


def test_martha_tower_is_martha_plus_the_posts_of_its_tower():
    m, t = asdict(ROBOTS['martha']), asdict(ROBOTS['martha_tower'])
    assert {k for k in m if m[k] != t[k]} == {'name', 'lidar_offset_x', 'posts'}
    assert ROBOTS['martha'].posts == () and t['lidar_offset_x'] == 0.19
    # 20 x 40 mm posts, the long side along x, at x = -0.033 and -0.208 m, 0.24 m between
    # outer faces across, inside the footprint (measured on the robot, 2026-10-21).
    xs = sorted({x for x, *_ in t['posts']})
    ys = sorted({y for _, y, *_ in t['posts']})
    assert len(t['posts']) == 4 and all((sx, sy) == (0.04, 0.02) for *_, sx, sy in t['posts'])
    assert xs == pytest.approx([-0.208, -0.033]) and ys[1] - ys[0] + 0.02 == pytest.approx(0.24)
    assert max(map(abs, xs)) + 0.02 < ROBOTS['martha_tower'].length / 2
    assert ys[1] + 0.01 < ROBOTS['martha_tower'].width / 2


def test_profiles_recorded_before_the_posts_still_load():
    """policies/martha.npz was exported before the posts field existed."""
    from pathlib import Path
    npz = Path(__file__).resolve().parents[1] / 'policies' / 'martha.npz'
    recorded = json.loads(str(np.load(npz)['settings']))['robot_profile']
    assert 'posts' not in recorded
    check_profile(recorded, 'martha', npz)
    with pytest.raises(ValueError, match=r'changed: name, lidar_offset_x, posts'):
        check_profile(recorded, 'martha_tower', npz)
    # And a profile with posts survives the JSON round trip of an export.
    check_profile(json.loads(json.dumps(asdict(ROBOTS['martha_tower']))), 'martha_tower', npz)
