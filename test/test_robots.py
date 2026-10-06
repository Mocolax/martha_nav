import json

import numpy as np
import yaml

from martha_nav.robots import ROBOTS, checkpoint_robot


def test_martha_keeps_the_numbers_the_code_used_before_the_profiles():
    m = ROBOTS['martha']
    assert (m.length, m.width, m.footprint_offset_x, m.lidar_offset_x) == (0.56, 0.41, 0.0, 0.2325)
    assert (m.lidar_range, m.lidar_min, m.lidar_rate) == (8.0, 0.0, 10.0)
    assert (m.v_max, m.v_reverse, m.v_lateral, m.w_max, m.inflation) == (0.35, 0.15, 0.25, 0.8, 0.40)
    assert m.holonomic


def test_burger_is_a_small_differential_robot():
    b = ROBOTS['burger']
    assert (b.length, b.width, b.footprint_offset_x, b.lidar_offset_x) == (0.14, 0.178, -0.032, -0.032)
    assert (b.v_max, b.w_max, b.inflation, b.lidar_rate) == (0.22, 1.5, 0.20, 5.0)
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
