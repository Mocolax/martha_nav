import xml.etree.ElementTree as ET

import pytest

from martha_nav.ros.world_speed import create_scaled_world
from martha_nav.sim2d.worlds import WORLDS_DIR

LAB = WORLDS_DIR / 'lab.world'


def physics(path):
    node = ET.parse(path).getroot().find('world').find('physics')
    return {child.tag: child.text for child in node}


def test_default_keeps_real_time(tmp_path):
    out = create_scaled_world(LAB, directory=tmp_path)
    values = physics(out)
    assert float(values['max_step_size']) == 0.001
    assert float(values['real_time_update_rate']) == 1000.0


def test_speed_factor_and_step_size(tmp_path):
    out = create_scaled_world(LAB, speed_factor=4.0, physics_step_size=0.002, directory=tmp_path)
    values = physics(out)
    assert float(values['max_step_size']) == 0.002
    assert float(values['real_time_update_rate']) == 2000.0      # 4 / 0.002
    assert float(values['real_time_factor']) == 4.0


def test_the_original_world_is_untouched(tmp_path):
    before = LAB.read_bytes()
    create_scaled_world(LAB, speed_factor=8.0, directory=tmp_path)
    assert LAB.read_bytes() == before


def test_rejects_out_of_range_values(tmp_path):
    with pytest.raises(ValueError):
        create_scaled_world(LAB, speed_factor=50.0, directory=tmp_path)
    with pytest.raises(ValueError):
        create_scaled_world(LAB, physics_step_size=0.5, directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        create_scaled_world(tmp_path / 'nope.world', directory=tmp_path)


def test_state_plugin_is_added_once(tmp_path):
    out = create_scaled_world(LAB, directory=tmp_path)
    world = ET.parse(out).getroot().find('world')
    plugins = [p for p in world.findall('plugin') if p.get('filename') == 'libgazebo_ros_state.so']
    assert len(plugins) == 1
    assert plugins[0].find('update_rate').text == '50.0'
    assert plugins[0].find('ros/namespace').text == '/gazebo'
    # Adding it again to an already-patched world keeps a single copy.
    again = create_scaled_world(out, directory=tmp_path)
    world = ET.parse(again).getroot().find('world')
    assert len([p for p in world.findall('plugin')
                if p.get('filename') == 'libgazebo_ros_state.so']) == 1
