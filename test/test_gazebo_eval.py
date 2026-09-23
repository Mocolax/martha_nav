"""Tests for the parts of gazebo_eval that do not need a running Gazebo."""
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


def test_the_stall_rule_matches_the_2d_environment():
    """gazebo_eval must cut a stuck episode like NavEnv does, not wait for the timeout."""
    import inspect

    from martha_nav.ros import gazebo_eval
    from martha_nav.sim2d.env import EnvConfig
    source = inspect.getsource(gazebo_eval.GazeboEval.__init__)
    assert "'no_progress_seconds', 15.0" in source
    assert EnvConfig().no_progress_time == 15.0
    assert "outcome = 'stalled'" in inspect.getsource(gazebo_eval.GazeboEval.run_episode)


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
