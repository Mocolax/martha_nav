import shutil
import subprocess
from pathlib import Path

import pytest
import xacro

from martha_nav.robots import ROBOTS

MARTHA = ROBOTS['martha']
LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH = MARTHA.lidar_offset_x, MARTHA.length, MARTHA.width

URDF = Path(__file__).resolve().parents[1] / 'urdf' / 'martha.urdf.xacro'


CONTROLLERS = Path(__file__).resolve().parents[1] / 'config' / 'controllers.yaml'


def robot(drive='mecanum'):
    mappings = {'drive': drive, 'controllers_file': str(CONTROLLERS)}
    return xacro.process_file(str(URDF), mappings=mappings).toprettyxml()


def joint_origin(doc, joint_name):
    """The xyz of a joint's <origin>, as three floats."""
    head = doc.split(f'name="{joint_name}"', 1)[1]
    xyz = head.split('<origin', 1)[1].split('xyz="', 1)[1].split('"', 1)[0]
    return [float(v) for v in xyz.split()]


def test_xacro_expands():
    assert '<robot' in robot() and 'xacro' not in robot().lower().split('<robot', 1)[1]


def contact_shell(doc):
    """The length and width of the contact shell box."""
    size = doc.split('name="contact_shell_collision"', 1)[1].split('size="', 1)[1].split('"', 1)[0]
    sx, sy, _ = (float(v) for v in size.split())
    return sx, sy


def test_contact_shell_is_the_2d_footprint_lengthened_to_cover_the_wheels():
    sx, sy = contact_shell(robot())
    assert sy == ROBOT_WIDTH
    assert ROBOT_LENGTH < sx <= ROBOT_LENGTH + 0.02     # Gazebo crashes at most 1 cm early


def test_chassis_and_wheels_keep_the_original_measurements():
    """martha/urdf/learning.xacro, the model of the real robot."""
    doc = robot()
    assert joint_origin(doc, 'base_front_left_wheel_joint') == [0.21, 0.175, 0.08]
    assert 'size="0.238 0.41 0.025"' in doc and doc.count('size="0.16 0.24 0.025"') == 2


def test_lidar_sits_where_the_observation_expects_it():
    assert abs(joint_origin(robot(), 'base_lidar')[0] - LIDAR_OFFSET_X) < 1e-9


def test_sensors_are_the_same_in_both_drives():
    for drive in ('mecanum', 'planar'):
        doc = robot(drive)
        for needle in ('libgazebo_ros_ray_sensor.so', 'libgazebo_ros_bumper.so',
                       'base_link_fixed_joint_lump__contact_shell_collision_collision'):
            assert needle in doc, (drive, needle)
        assert doc.count('<sensor ') == 2             # ray + contact, nothing else


def test_mecanum_urdf_parses_as_a_parameter_override_rule():
    """gazebo_ros2_control hands the whole URDF to rcl as "robot_description:=<urdf>",
    and that value is parsed as YAML: a colon plus a space, or a remapping rule,
    anywhere inside it (comments included) stops the controller_manager from starting."""
    doc = robot('mecanum')
    assert ': ' not in doc
    assert ':=' not in doc


def test_mecanum_drive_is_the_default():
    doc = robot()
    assert '<ros2_control' in doc and 'libgazebo_ros2_control.so' in doc
    assert 'controllers.yaml' in doc
    assert doc.count('_roller_') > 0
    assert doc.count('type="continuous"') == 52       # 4 wheels + 48 rollers
    assert 'libgazebo_ros_planar_move.so' not in doc


def test_planar_drive_swaps_the_wheels_for_the_plugin():
    doc = robot('planar')
    assert 'libgazebo_ros_planar_move.so' in doc and 'cmd_vel:=/cmd_vel' in doc
    assert '<ros2_control' not in doc
    assert '_roller_' not in doc
    assert doc.count('type="continuous"') == 0


def test_dropped_parts_of_the_old_urdf_are_gone():
    # Tags and link names, not words: the header comment mentions what was dropped.
    doc = robot()
    for absent in ('imu_link', 'imu_sensor', '${robot_namespace}', '${frame_prefix}'):
        assert absent not in doc, absent


def test_lidar_scans_the_full_circle_at_8_m():
    doc = robot()
    ray = doc.split('name="lidar_sensor"', 1)[1]
    assert '<samples>360</samples>' in ray
    assert '<max>8.0</max>' in ray
    assert '<update_rate>10</update_rate>' in ray


def test_wheels_stay_inside_the_contact_shell():
    """A wall must touch the shell (which the bumper watches) before a wheel."""
    for drive, reach in (('mecanum', 0.079), ('planar', 0.075)):  # rollers, fixed wheel
        doc = robot(drive)
        x, y, _ = joint_origin(doc, 'base_front_left_wheel_joint')
        sx, sy = contact_shell(doc)
        assert x + reach < sx / 2, drive
        assert y + 0.045 / 2 < sy / 2, drive          # 0.045 m is the wheel width


BURGER_URDF = Path(__file__).resolve().parents[1] / 'urdf' / 'burger.urdf.xacro'
BURGER = ROBOTS['burger']


def burger():
    return xacro.process_file(str(BURGER_URDF)).toprettyxml()


def test_burger_urdf_parses_as_a_parameter_override_rule():
    """Martha's rule: the URDF can travel inside a "robot_description:=<urdf>" override."""
    doc = burger()
    assert ': ' not in doc
    assert ':=' not in doc


def test_burger_contact_shell_is_the_2d_footprint():
    doc = burger()
    size = doc.split('name="contact_shell_collision"', 1)[1].split('size="', 1)[1].split('"', 1)[0]
    sx, sy, _ = (float(v) for v in size.split())
    assert (sx, sy) == (BURGER.length, BURGER.width)
    assert joint_origin(doc, 'contact_shell_joint')[0] == BURGER.footprint_offset_x


def test_burger_lidar_matches_the_profile():
    doc = burger()
    assert joint_origin(doc, 'scan_joint')[0] == BURGER.lidar_offset_x
    ray = doc.split('<range>', 1)[1]
    assert float(ray.split('<min>', 1)[1].split('<', 1)[0]) == BURGER.lidar_min
    assert float(ray.split('<max>', 1)[1].split('<', 1)[0]) == BURGER.lidar_range
    rate = doc.split('name="lidar_sensor"', 1)[1].split('<update_rate>', 1)[1].split('<', 1)[0]
    assert float(rate) == BURGER.lidar_rate


@pytest.mark.skipif(shutil.which('gz') is None, reason='needs Gazebo')
def test_burger_contact_sensor_watches_a_collision_gazebo_keeps(tmp_path):
    """Gazebo renames collisions when it lumps fixed joints; the sensor must use the new name."""
    path = tmp_path / 'burger.urdf'
    path.write_text(burger())
    sdf = subprocess.run(['gz', 'sdf', '-p', str(path)], capture_output=True, text=True,
                         check=True).stdout
    watched = sdf.split('<contact>', 1)[1].split('<collision>', 1)[1].split('</collision>', 1)[0]
    assert 'contact_shell' in watched
    assert f"<collision name='{watched}'>" in sdf or f'<collision name="{watched}">' in sdf
