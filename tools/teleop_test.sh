#!/usr/bin/env bash
# Teleop on the real robot without encoders (firmware/motor_test): ESP32 bridge, RPLIDAR,
# the robot model and RViz with the scan around base_link. Drive from another terminal
# with teleop_twist_keyboard. Ctrl-C stops everything.
ESP32=/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
LIDAR=/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_9d653b21a3e51e42bf6b7027b0d61dd1-if00-port0

docker start ros2_humble >/dev/null
exec docker exec -it -e XAUTHORITY="$XAUTHORITY" -e ESP32="$ESP32" -e LIDAR="$LIDAR" \
  -w /home/ros/ros2_ws/src/martha_nav ros2_humble bash -lc '
source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash
trap "kill 0" EXIT
xacro urdf/martha.urdf.xacro drive:=planar tower:=true > /tmp/martha.urdf
ros2 run robot_state_publisher robot_state_publisher /tmp/martha.urdf &
ros2 run martha_nav esp32_bridge --ros-args -p port:=$ESP32 &
ros2 run rplidar_ros rplidar_node --ros-args --params-file config/rplidar.yaml -p serial_port:=$LIDAR &
rviz2 -d rviz/nav.rviz -f base_link --ros-args -r /gazebo_ros_lidar/out:=/scan &
wait'
