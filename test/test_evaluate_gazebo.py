"""Tests for the parts of evaluate_gazebo that do not need a running Gazebo."""
import pytest

from martha_nav.ros.evaluate_gazebo import BOX_SDF, CYLINDER_SDF
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
    from martha_nav.ros.evaluate_gazebo import episode_outcome
    from martha_nav.sim2d.dynamics import DT
    from martha_nav.sim2d.env import episode_steps
    from martha_nav.sim2d.planner import Path, RouteProgress
    route = Path([[0.0, 0.0], [4.0, 0.0]])
    limit = episode_steps(route.length) * DT
    moving, stuck = RouteProgress(route), RouteProgress(route)
    moving.update(1.0, 0.0)
    for _ in range(150):                               # 15 s pacing on the spot
        stuck.update(0.0, 0.0)
    assert episode_outcome(True, 'succeeded', 0.2, stuck, limit, limit, 15.0) == 'collision'
    assert episode_outcome(False, 'succeeded', 0.2, stuck, limit, limit, 15.0) == 'success'
    assert episode_outcome(False, 'failed', 3.0, moving, 1.0, limit, 15.0) == 'failed'
    assert episode_outcome(False, 'active', 3.0, stuck, limit, limit, 15.0) == 'stalled'
    assert episode_outcome(False, 'active', 3.0, moving, limit, limit, 15.0) == 'timeout'
    assert episode_outcome(False, 'active', 3.0, moving, limit - 0.1, limit, 15.0) is None


def test_a_goal_reached_only_by_the_localization_is_lost_not_a_success():
    """Under SLAM the planner says 'succeeded' from the estimated pose; the true pose decides."""
    from martha_nav.ros.evaluate_gazebo import episode_outcome
    from martha_nav.sim2d.planner import Path, RouteProgress
    progress = RouteProgress(Path([[0.0, 0.0], [4.0, 0.0]]))
    assert episode_outcome(False, 'succeeded', 0.45, progress, 10.0, 60.0, 15.0) == 'success'
    assert episode_outcome(False, 'succeeded', 2.0, progress, 10.0, 60.0, 15.0) == 'lost'


def test_shards_split_the_seeds_between_parallel_simulations():
    """shard:=i/n plays every n-th seed from the i-th, so n Gazebos cover the set once."""
    import rclpy

    from martha_nav.ros.evaluate_gazebo import EvaluateGazebo
    from martha_nav.sim2d.env import eval_seeds
    played = []
    for i in range(3):
        rclpy.init(args=['--ros-args', '-p', f'shard:={i}/3', '-p', 'episodes:=10'])
        node = EvaluateGazebo()
        played += node.seeds
        node.destroy_node()
        rclpy.try_shutdown()
    assert sorted(played) == eval_seeds(10)


@pytest.fixture
def evaluate_gazebo_node():
    import rclpy

    from martha_nav.ros.evaluate_gazebo import EvaluateGazebo
    rclpy.init()
    node = EvaluateGazebo()
    yield node
    node.destroy_node()
    rclpy.try_shutdown()


def test_episode_time_is_simulated_time(evaluate_gazebo_node):
    """Gazebo rarely runs at exactly 1x; the rules are in seconds of simulation, as in 2D."""
    assert evaluate_gazebo_node.get_parameter('use_sim_time').value is True


def test_the_goal_is_cancelled_before_the_teleport(evaluate_gazebo_node):
    """Otherwise the planner resumes the previous goal and the robot leaves the start."""
    from martha_nav.sim2d.scenarios import generate
    node, calls = evaluate_gazebo_node, []
    for name in ('cancel_goal', 'set_initial_pose', 'spawn_obstacles',
                 'spawn_goal_marker', 'clear_obstacles', 'spin'):
        setattr(node, name, lambda *a, name=name: calls.append(name) or [])
    node.send_goal = lambda goal: setattr(node, 'status', 'failed')
    node.position = (99.0, 99.0)                       # where the last episode left it
    node.teleport = lambda x, y, yaw: calls.append('teleport')
    node.spin = lambda seconds: (calls.append('spin'), 'teleport' in calls
                                 and setattr(node, 'position', (x0, y0)))
    x0, y0, _ = generate(node.seeds[0], node.cfg).start
    row = node.run_episode(node.seeds[0])
    assert row['outcome'] == 'failed'
    assert calls.index('cancel_goal') < calls.index('teleport')
    # A teleport is a kidnapping for a localizer: tell it where the robot is, as an
    # operator does in RViz before the demo.
    assert calls.index('teleport') < calls.index('set_initial_pose')
    # ...but only once Gazebo shows it there, or the localizer reads the jump as odometry.
    assert 'spin' in calls[calls.index('teleport'):calls.index('set_initial_pose')]
    assert {'loc_err_mean', 'loc_err_max', 'yaw_err_max_deg', 'travelled', 'spl'} <= set(row)
    assert row['spl'] == 0.0 and isinstance(row['trajectory'], list)


def test_goal_marker_has_no_collision():
    """The marker is decoration: a collision would show up in the LiDAR."""
    import xml.etree.ElementTree as ET

    from martha_nav.ros.evaluate_gazebo import GOAL_SDF
    model = ET.fromstring(GOAL_SDF).find('model')
    assert model.get('name') == 'goal_marker'
    assert model.find('link/collision') is None
    assert len(model.findall('link/visual')) == 2


def test_position_comes_from_gazebo_not_from_odometry():
    """The odometry topic depends on the drive; model_states always exists."""
    import inspect

    from martha_nav.ros import evaluate_gazebo
    source = inspect.getsource(evaluate_gazebo.EvaluateGazebo)
    assert '/gazebo/model_states' in source
    assert "'/odom'" not in source


class FakeClient:
    srv_name = '/spawn_entity'

    def __init__(self, available=True, success=True):
        self.available, self.success = available, success

    def wait_for_service(self, timeout_sec=None):
        return self.available

    def call_async(self, request):
        from rclpy.task import Future
        response = type('Response', (), {'success': self.success, 'status_message': 'exists'})()
        future = Future()
        future.set_result(response)
        return future


def test_a_failed_gazebo_call_stops_the_evaluation(evaluate_gazebo_node):
    """A silent failure (an obstacle left by an interrupted run) would corrupt the results."""
    evaluate_gazebo_node.call(FakeClient(), None)
    with pytest.raises(RuntimeError, match='exists'):
        evaluate_gazebo_node.call(FakeClient(success=False), None)
    with pytest.raises(RuntimeError, match='not available'):
        evaluate_gazebo_node.call(FakeClient(available=False), None)


def test_localization_error_is_distance_and_wrapped_yaw():
    import math

    from martha_nav.ros.evaluate_gazebo import localization_error
    distance, yaw = localization_error((0.0, 0.0, 3.1), (0.3, 0.4, -3.1))
    assert math.isclose(distance, 0.5)
    assert math.isclose(yaw, 2 * math.pi - 6.2)
