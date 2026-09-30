"""Tests for the parts of gazebo_eval that do not need a running Gazebo."""
import pytest

from martha_nav.ros.gazebo_eval import BOX_SDF, CYLINDER_SDF
from martha_nav.sim2d.scenarios import Obstacle


def test_box_sdf_is_well_formed_and_uses_the_size():
    import xml.etree.ElementTree as ET
    ob = Obstacle('box', 1.0, 2.0, sx=0.4, sy=0.25, yaw=0.3)
    xml = BOX_SDF.format(name='obstacle_0', sx=ob.sx, sy=ob.sy)
    size = ET.fromstring(xml).find('model/link/collision/geometry/box/size').text
    assert size.startswith('0.4 0.25')
    assert ET.fromstring(xml).find('model').get('name') == 'obstacle_0'


def test_cylinder_sdf_is_well_formed_and_uses_the_radius():
    import xml.etree.ElementTree as ET
    xml = CYLINDER_SDF.format(name='obstacle_1', r=0.17)
    radius = ET.fromstring(xml).find('model/link/collision/geometry/cylinder/radius').text
    assert float(radius) == 0.17
    assert ET.fromstring(xml).find('model').get('name') == 'obstacle_1'


def test_episodes_end_by_the_2d_rules_in_their_order():
    """Collision, then success, then a stall on the route, then the route-length timeout."""
    from martha_nav.ros.gazebo_eval import episode_outcome
    from martha_nav.sim2d.dynamics import DT
    from martha_nav.sim2d.env import episode_steps
    from martha_nav.sim2d.planner import Path, RouteProgress
    route = Path([[0.0, 0.0], [4.0, 0.0]])
    limit = episode_steps(route.length) * DT
    moving, stuck = RouteProgress(route), RouteProgress(route)
    moving.update(1.0, 0.0)
    for _ in range(150):                               # 15 s pacing on the spot
        stuck.update(0.0, 0.0)
    assert episode_outcome(True, 'succeeded', stuck, limit, limit, 15.0) == 'collision'
    assert episode_outcome(False, 'succeeded', stuck, limit, limit, 15.0) == 'success'
    assert episode_outcome(False, 'failed', moving, 1.0, limit, 15.0) == 'failed'
    assert episode_outcome(False, 'active', stuck, limit, limit, 15.0) == 'stalled'
    assert episode_outcome(False, 'active', moving, limit, limit, 15.0) == 'timeout'
    assert episode_outcome(False, 'active', moving, limit - 0.1, limit, 15.0) is None


@pytest.fixture
def gazebo_eval_node():
    import rclpy

    from martha_nav.ros.gazebo_eval import GazeboEval
    rclpy.init()
    node = GazeboEval()
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def test_episode_time_is_simulated_time(gazebo_eval_node):
    """Gazebo rarely runs at exactly 1x; the rules are in seconds of simulation, as in 2D."""
    assert gazebo_eval_node.get_parameter('use_sim_time').value is True


def test_the_goal_is_cancelled_before_the_teleport(gazebo_eval_node):
    """Otherwise the planner resumes the previous goal and the robot leaves the start."""
    node, calls = gazebo_eval_node, []
    for name in ('cancel_goal', 'teleport', 'spawn_obstacles', 'spawn_goal_marker',
                 'clear_obstacles', 'spin'):
        setattr(node, name, lambda *a, name=name: calls.append(name) or [])
    node.send_goal = lambda goal: setattr(node, 'status', 'failed')
    assert node.run_episode(node.seeds[0])['outcome'] == 'failed'
    assert calls.index('cancel_goal') < calls.index('teleport')


def test_goal_marker_has_no_collision():
    """The marker is decoration: a collision would show up in the LiDAR."""
    import xml.etree.ElementTree as ET

    from martha_nav.ros.gazebo_eval import GOAL_SDF
    model = ET.fromstring(GOAL_SDF).find('model')
    assert model.get('name') == 'goal_marker'
    assert model.find('link/collision') is None
    assert len(model.findall('link/visual')) == 2


def test_position_comes_from_gazebo_not_from_odometry():
    """The odometry topic depends on the drive; model_states always exists."""
    import inspect

    from martha_nav.ros import gazebo_eval
    source = inspect.getsource(gazebo_eval.GazeboEval)
    assert '/gazebo/model_states' in source
    assert "'/odom'" not in source
