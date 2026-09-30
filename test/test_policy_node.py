"""policy_node's handling of /plan and /nav_status, with a stand-in model."""
import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from std_msgs.msg import String

from martha_nav.ros import policy_node


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
    monkeypatch.setattr(policy_node, 'load_model', lambda path: FakeModel())
    monkeypatch.setattr(policy_node, '_trained_env', lambda path: {})
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=/fake/best_model.zip'])
    node = policy_node.PolicyNode()
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
