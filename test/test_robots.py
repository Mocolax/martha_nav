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
    assert (b.v_max, b.w_max, b.inflation, b.lidar_rate) == (0.22, 1.5, 0.30, 5.0)
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
        check_profile({k: v for k, v in burger.items() if k != 'inflation'}, 'burger',
                      'best_model.zip')


@pytest.mark.parametrize('name', list(ROBOTS))
def test_the_guard_reaches_beyond_the_lidars_blind_zone(name):
    """An obstacle that has just left the blind zone must already be inside the guard."""
    r = ROBOTS[name]
    reach = r.length / 2 + r.footprint_offset_x - r.lidar_offset_x + r.guard_margin
    assert 'front' in footprint_blocked([reach - 1e-6], [0.0], r)
    assert 'front' not in footprint_blocked([reach + 1e-3], [0.0], r)
    assert reach - r.lidar_min >= 0.04
