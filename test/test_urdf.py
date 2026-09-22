from pathlib import Path

import xacro

from martha_nav.sim2d.geometry import LIDAR_OFFSET_X, ROBOT_LENGTH, ROBOT_WIDTH

URDF = Path(__file__).resolve().parents[1] / 'urdf' / 'martha.urdf.xacro'


def robot():
    return xacro.process_file(str(URDF)).toprettyxml()


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


def test_required_plugins_and_remappings():
    doc = robot()
    for needle in ('libgazebo_ros_planar_move.so', 'libgazebo_ros_ray_sensor.so',
                   'libgazebo_ros_bumper.so', '~/out:=/scan', 'cmd_vel:=/cmd_vel',
                   'bumper_states:=/bumper_states',
                   'base_link_fixed_joint_lump__contact_shell_collision_collision'):
        assert needle in doc, needle
    assert doc.count('<sensor ') == 2                 # ray + contact, nothing else


def test_dropped_parts_of_the_old_urdf_are_gone():
    # Tags and link names, not words: the header comment mentions what was dropped.
    doc = robot()
    for absent in ('<ros2_control', 'imu_link', 'imu_sensor', '_roller_',
                   'gazebo_ros2_control', '${robot_namespace}', '${frame_prefix}'):
        assert absent not in doc, absent


def test_lidar_scans_the_full_circle_at_8_m():
    doc = robot()
    ray = doc.split('name="lidar_sensor"', 1)[1]
    assert '<samples>360</samples>' in ray
    assert '<max>8.0</max>' in ray
    assert '<update_rate>10</update_rate>' in ray


def test_wheels_stay_inside_the_contact_shell():
    """A wall must touch the shell (which the bumper watches) before a wheel."""
    doc = robot()
    radius = float(doc.split('name="front_left_wheel"', 1)[1].split('radius="', 1)[1].split('"', 1)[0])
    x = joint_origin(doc, 'base_front_left_wheel_joint')[0]
    assert x + radius < ROBOT_LENGTH / 2
