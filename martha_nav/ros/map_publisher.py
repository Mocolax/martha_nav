"""Publish a rasterised .world as a latched /map."""
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from martha_nav.ros.occupancy import grid_to_msg
from martha_nav.sim2d.worlds import rasterize_world

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        world = self.declare_parameter('world', 'lab').value
        frame = self.declare_parameter('frame_id', 'map').value
        msg = grid_to_msg(rasterize_world(world), frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.pub.publish(msg)
        self.get_logger().info(
            f'published {world}: {msg.info.width}x{msg.info.height} cells @ {msg.info.resolution} m')


def main():
    rclpy.init()
    node = MapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
