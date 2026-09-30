"""Publish map -> odom from Gazebo's true pose, so map -> base_link is exact.

Evaluation in Gazebo measures the policy, not the localisation (spec 7.2).
"""
import math

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from martha_nav.ros.common import run_node, yaw_of


class GazeboGroundTruthTf(Node):
    def __init__(self):
        super().__init__('gazebo_ground_truth_tf')
        self.model = self.declare_parameter('model_name', 'martha').value
        self.truth = None                      # (x, y, yaw) of base_link in map
        self.broadcaster = TransformBroadcaster(self)
        # Off when slam_toolbox localizes: then it owns map -> odom.
        if self.declare_parameter('publish_map_odom', True).value:
            self.create_subscription(ModelStates, '/gazebo/model_states', self.on_states, 10)
            self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        # The mecanum controller publishes odom -> base_link on its own topic instead
        # of /tf, and it lives inside gzserver, so it cannot be remapped from the launch.
        relay = self.declare_parameter('odom_tf_topic', '').value
        if relay:
            self.create_subscription(TFMessage, relay, self.on_odom_tf, 10)
            self.get_logger().info(f'relaying {relay} to /tf')

    def on_states(self, msg):
        if self.model not in msg.name:
            return
        pose = msg.pose[msg.name.index(self.model)]
        self.truth = (pose.position.x, pose.position.y, yaw_of(pose.orientation))

    def on_odom_tf(self, msg):
        for transform in msg.transforms:
            self.broadcaster.sendTransform(transform)

    def on_odom(self, msg):
        """map -> odom = (map -> base) * inverse(odom -> base), in the plane."""
        if self.truth is None:
            return
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        oyaw = yaw_of(msg.pose.pose.orientation)
        tx, ty, tyaw = self.truth
        dyaw = tyaw - oyaw
        c, s = math.cos(dyaw), math.sin(dyaw)
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id = 'odom'
        tf.transform.translation.x = tx - (c * ox - s * oy)
        tf.transform.translation.y = ty - (s * ox + c * oy)
        tf.transform.rotation.z = math.sin(dyaw / 2)
        tf.transform.rotation.w = math.cos(dyaw / 2)
        self.broadcaster.sendTransform(tf)


def main():
    run_node(GazeboGroundTruthTf)


if __name__ == '__main__':
    main()
