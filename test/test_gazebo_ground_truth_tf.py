import rclpy

from martha_nav.ros.gazebo_ground_truth_tf import GazeboGroundTruthTf

RELAY = '/mecanum_drive_controller/tf_odometry'


def subscribed(*params):
    rclpy.init(args=['--ros-args', '-p', f'odom_tf_topic:={RELAY}', *params])
    try:
        node = GazeboGroundTruthTf()
        topics = {s.topic_name for s in node.subscriptions}
        node.destroy_node()
        return topics
    finally:
        rclpy.try_shutdown()


def test_by_default_it_publishes_map_to_odom_from_the_true_pose():
    assert subscribed() == {'/gazebo/model_states', '/odom', RELAY}


def test_with_slam_it_only_relays_the_odometry_tf():
    """slam_toolbox owns map -> odom then; two publishers would fight over it."""
    assert subscribed('-p', 'publish_map_odom:=false') == {RELAY}
