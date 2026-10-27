from pathlib import Path

import pytest
import xacro

from martha_nav.robots import ROBOTS

MARTHA = ROBOTS['martha']
LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH = MARTHA.lidar_offset_x, MARTHA.length, MARTHA.width

URDF = Path(__file__).resolve().parents[1] / 'urdf' / 'martha.urdf.xacro'


CONTROLLERS = Path(__file__).resolve().parents[1] / 'config' / 'controllers.yaml'


def robot(drive='mecanum', tower='true'):
    mappings = {'drive': drive, 'controllers_file': str(CONTROLLERS), 'tower': tower}
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
    """martha/urdf/learning.xacro, the model ROBOTS['martha'] was trained for."""
    doc = robot(tower='false')
    assert joint_origin(doc, 'base_front_left_wheel_joint') == [0.21, 0.175, 0.08]
    assert 'size="0.238 0.41 0.025"' in doc and doc.count('size="0.16 0.24 0.025"') == 2


def test_the_robot_as_built_has_two_57_cm_boards_10_cm_apart():
    doc = robot()
    assert joint_origin(doc, 'base_front_left_wheel_joint') == [0.21, 0.175, 0.08]
    # The lower board keeps the chassis shape; the upper one is a plain rectangle.
    assert doc.count('size="0.25 0.41 0.005"') == 1 and doc.count('size="0.16 0.24 0.005"') == 2
    assert doc.count('size="0.57 0.405 0.005"') == 1
    assert 'size="0.238 0.41 0.025"' not in doc


def test_lidar_sits_where_the_observation_expects_it():
    assert abs(joint_origin(robot(tower='false'), 'base_lidar')[0] - LIDAR_OFFSET_X) < 1e-9
    tower = ROBOTS['martha_tower'].lidar_offset_x
    assert abs(joint_origin(robot(), 'base_lidar')[0] - tower) < 1e-9


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


def box(doc, link):
    """Centre (from the link's joint) and size of a one-box link."""
    size = doc.split(f'<link name="{link}">', 1)[1].split('size="', 1)[1].split('"', 1)[0]
    return joint_origin(doc, f'base_{link}_joint'), [float(v) for v in size.split()]


def test_the_tower_posts_are_the_ones_the_2d_lidar_sees():
    doc = robot()
    posts = sorted((x, y, sx, sy) for (x, y, _), (sx, sy, _) in (
        box(doc, f'tower_post_{end}_{side}') for end in ('front', 'rear')
        for side in ('left', 'right')))
    assert posts == pytest.approx(sorted(ROBOTS['martha_tower'].posts))


def test_the_tower_posts_cross_the_scan_plane_and_rise_0_91_m_from_the_upper_board():
    doc = robot()
    (_, _, z), (_, _, height) = box(doc, 'tower_post_front_left')
    scan_z = joint_origin(doc, 'base_lidar')[2]
    assert height == 0.91 and z - height / 2 == pytest.approx(0.20)
    assert z - height / 2 < scan_z < z + height / 2


def test_the_lidar_scans_3_1_cm_above_the_upper_board_and_under_the_crossbars():
    doc = robot()
    scan_z = joint_origin(doc, 'base_lidar')[2]
    assert scan_z == pytest.approx(0.231)
    for bar in ('front_low', 'rear_low'):
        (_, _, z), (_, _, sz) = box(doc, f'tower_crossbar_{bar}')
        assert z - sz / 2 == pytest.approx(0.20 + 0.065) and z - sz / 2 > scan_z


def test_the_tower_stays_inside_the_2d_footprint():
    doc = robot()
    links = [name.split('"', 1)[0] for name in doc.split('<link name="')[1:]]
    for link in (n for n in links if n.startswith('tower_')):
        (x, y, _), (sx, sy, _) = box(doc, link)
        assert abs(x) + sx / 2 <= ROBOT_LENGTH / 2 and abs(y) + sy / 2 <= ROBOT_WIDTH / 2, link


def test_without_the_tower_the_model_is_the_trained_martha():
    for drive in ('mecanum', 'planar'):
        assert 'tower_' not in robot(drive, tower='false')
        assert 'tower_post_front_left' in robot(drive)


