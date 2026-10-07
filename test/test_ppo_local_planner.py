"""ppo_local_planner's handling of /plan and /nav_status, with a stand-in model."""
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from std_msgs.msg import String

from martha_nav.ros import ppo_local_planner
from martha_nav.ros.numpy_policy import settings_of
from martha_nav.sim2d.env import EnvConfig


class FakeModel:
    def predict(self, obs, deterministic=True):
        return np.zeros(2), None


def plan(*points):
    msg = PathMsg()
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        msg.poses.append(pose)
    return msg


@pytest.fixture
def node(monkeypatch):
    monkeypatch.setattr(ppo_local_planner, 'load_policy',
                        lambda path: (FakeModel(), settings_of(EnvConfig())))
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/best_model.zip'])
    node = ppo_local_planner.PpoLocalPlanner()
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def test_a_replanned_route_does_not_reset_the_policy(node):
    node.on_status(String(data='active'))
    node.on_plan(plan((0, 0), (5, 0)))
    node.core.lstm_state = 'memory'
    node.on_plan(plan((0, 1), (2, 1), (5, 0)))
    assert node.core.lstm_state == 'memory'
    assert np.allclose(node.path.points[0], [0.0, 1.0])


def test_the_route_is_forgotten_when_the_episode_ends(node):
    """A stale route must not be driven when the next goal turns the status active."""
    node.on_status(String(data='active'))
    node.on_plan(plan((0, 0), (5, 0)))
    node.core.lstm_state = 'memory'
    node.on_status(String(data='succeeded'))
    assert node.path is None
    assert node.core.lstm_state is None


def test_a_checkpoint_for_another_robot_is_refused(monkeypatch):
    monkeypatch.setattr(ppo_local_planner, 'load_policy',
                        lambda path: (FakeModel(), settings_of(EnvConfig(robot='burger'))))
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/policy.npz', '-p', 'robot:=martha'])
    try:
        with pytest.raises(RuntimeError, match='burger'):
            ppo_local_planner.PpoLocalPlanner()
    finally:
        rclpy.try_shutdown()


def test_the_core_drives_the_models_robot(monkeypatch):
    monkeypatch.setattr(ppo_local_planner, 'load_policy',
                        lambda path: (FakeModel(), settings_of(EnvConfig(robot='burger'))))
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/policy.npz'])
    try:
        node = ppo_local_planner.PpoLocalPlanner()
        assert node.core.robot.name == 'burger'
        node.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_speed_scale_slows_every_velocity():
    cmd = ppo_local_planner.to_twist((0.2, 1.0), 0.5)
    assert (cmd.linear.x, cmd.linear.y, cmd.angular.z) == (0.1, 0.0, 0.5)
    cmd = ppo_local_planner.to_twist((0.2, -0.1, 1.0))
    assert (cmd.linear.x, cmd.linear.y, cmd.angular.z) == (0.2, -0.1, 1.0)


@pytest.mark.parametrize('scale', ['-0.5', '0.0', '1.5'])
def test_a_speed_scale_outside_zero_to_one_is_refused(monkeypatch, scale):
    monkeypatch.setattr(ppo_local_planner, 'load_policy',
                        lambda path: (FakeModel(), settings_of(EnvConfig())))
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/best_model.zip',
                     '-p', f'speed_scale:={scale}'])
    try:
        with pytest.raises(RuntimeError, match='speed_scale'):
            ppo_local_planner.PpoLocalPlanner()
    finally:
        rclpy.try_shutdown()
