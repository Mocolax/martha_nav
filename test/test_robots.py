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


def test_burger_is_a_small_differential_robot():
    b = ROBOTS['burger']
    assert (b.length, b.width, b.footprint_offset_x, b.lidar_offset_x) == (0.14, 0.178, -0.032, -0.032)
    assert (b.v_max, b.w_max, b.inflation, b.lidar_rate) == (0.22, 1.5, 0.40, 5.0)
    assert b.guard_margin == 0.10
    assert not b.holonomic


def test_a_checkpoint_says_which_robot_it_drives(tmp_path):
    model = tmp_path / 'best_model.zip'
    assert checkpoint_robot(model) == 'martha'                          # no config.yaml
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'n_rays': 180}}))
    assert checkpoint_robot(model) == 'martha'                          # trained before profiles
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'env': {'robot': 'burger'}}))
    assert checkpoint_robot(model) == 'burger'
    exported = tmp_path / 'policy.npz'
    np.savez(exported, settings=json.dumps({'robot': 'burger'}))
    assert checkpoint_robot(exported) == 'burger'


def test_a_recorded_profile_must_be_the_current_one():
    burger = asdict(ROBOTS['burger'])
    check_profile(burger, 'burger', 'best_model.zip')
    check_profile(None, 'burger', 'best_model.zip')                     # trained before profiles
    changed = {**burger, 'lidar_min': 0.16, 'guard_margin': 0.15}
    with pytest.raises(ValueError, match=r'different burger profile \(changed: lidar_min, '
                                         r'guard_margin\): retrain and re-export'):
        check_profile(changed, 'burger', 'best_model.zip')
    with pytest.raises(ValueError, match='best_model.zip'):
        check_profile({k: v for k, v in burger.items() if k != 'lidar_min'}, 'burger',
                      'best_model.zip')


def test_the_inflation_is_the_planners_not_the_models():
    """A model trained with other routes still drives on the current planner's."""
    burger = asdict(ROBOTS['burger'])
    check_profile({**burger, 'inflation': 0.30}, 'burger', 'best_model.zip')
    check_profile({k: v for k, v in burger.items() if k != 'inflation'}, 'burger', 'best_model.zip')


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
    assert {k for k in m if m[k] != t[k]} == {'name', 'posts'}
    assert ROBOTS['martha'].posts == ()
    # 20 x 20 mm posts, 0.28 x 0.24 m between outer faces, inside the footprint.
    xs = sorted({x for x, _, _ in t['posts']})
    ys = sorted({y for _, y, _ in t['posts']})
    assert len(t['posts']) == 4 and all(side == 0.02 for *_, side in t['posts'])
    assert xs[1] - xs[0] + 0.02 == pytest.approx(0.28) and ys[1] - ys[0] + 0.02 == pytest.approx(0.24)
    assert xs[1] + 0.01 < ROBOTS['martha_tower'].length / 2
    assert ys[1] + 0.01 < ROBOTS['martha_tower'].width / 2


def test_profiles_recorded_before_the_posts_still_load():
    """policies/martha.npz was exported before the posts field existed."""
    from pathlib import Path
    npz = Path(__file__).resolve().parents[1] / 'policies' / 'martha.npz'
    recorded = json.loads(str(np.load(npz)['settings']))['robot_profile']
    assert 'posts' not in recorded
    check_profile(recorded, 'martha', npz)
    with pytest.raises(ValueError, match=r'changed: name, posts'):
        check_profile(recorded, 'martha_tower', npz)
    # And a profile with posts survives the JSON round trip of an export.
    check_profile(json.loads(json.dumps(asdict(ROBOTS['martha_tower']))), 'martha_tower', npz)
