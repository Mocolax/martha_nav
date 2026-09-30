"""planner_node's publishing order and goal cancellation, without TF or a map topic."""
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty

from martha_nav.ros.planner_node import PlannerNode
from martha_nav.sim2d.geometry import empty_grid


class Recorder:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def publish(self, msg):
        self.log.append((self.name, msg))


@pytest.fixture
def node():
    rclpy.init()
    node = PlannerNode()
    node.core.set_map(empty_grid(8.0, 4.0))
    node.pose = lambda: (1.0, 2.0)
    node.log = []
    node.plan_pub, node.status_pub = Recorder(node.log, 'plan'), Recorder(node.log, 'status')
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def goal(x, y):
    msg = PoseStamped()
    msg.pose.position.x, msg.pose.position.y = float(x), float(y)
    return msg


def test_the_route_goes_out_before_the_active_status(node):
    """Otherwise the policy sees 'active' while it still holds the previous route."""
    node.on_goal(goal(7.0, 2.0))
    node.tick()
    assert [name for name, _ in node.log] == ['plan', 'status']
    assert node.log[1][1].data == 'active'


def test_cancel_clears_the_route_and_reports_idle(node):
    node.on_goal(goal(7.0, 2.0))
    node.tick()
    node.log.clear()
    node.on_cancel(Empty())
    assert [name for name, _ in node.log] == ['plan', 'status']
    assert node.log[0][1].poses == [] and node.log[1][1].data == 'idle'
    node.tick()
    assert len(node.log) == 2                          # idle already reported: nothing new


def test_every_cancel_is_acknowledged(node):
    """gazebo_eval waits for 'idle' after each cancel, even when nothing was running."""
    node.on_cancel(Empty())
    node.on_cancel(Empty())
    assert [msg.data for name, msg in node.log if name == 'status'] == ['idle', 'idle']
