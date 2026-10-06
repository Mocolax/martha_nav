from pathlib import Path

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


def test_contact_shell_matches_the_2d_footprint():
    doc = robot()
    size = doc.split('name="contact_shell_collision"', 1)[1].split('size="', 1)[1].split('"', 1)[0]
    sx, sy, _ = (float(v) for v in size.split())
    assert (sx, sy) == (ROBOT_LENGTH, ROBOT_WIDTH)


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
    for drive in ('mecanum', 'planar'):
        doc = robot(drive)
        x = joint_origin(doc, 'base_front_left_wheel_joint')[0]
        assert x + 0.075 < ROBOT_LENGTH / 2, drive     # 0.075 m is the wheel envelope
