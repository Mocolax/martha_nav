#!/usr/bin/env bash
# Martha on the floor, from the laptop wired to her ESP32 and RPLIDAR, in the ros2_humble container.
#
#   ./robot.sh mapping            base + LiDAR + EKF + slam_toolbox mapping + RViz
#   ./robot.sh save-map <name>    (another terminal) save that map to maps/<name>
#   ./robot.sh <map>              base + LiDAR + EKF + localization in maps/<map> + RViz;
#                                 2D Pose Estimate if Martha is not where its mapping began
#   ./robot.sh policy [npz]       (another terminal, after ./robot.sh <map>) planners + PPO,
#                                 goals with 2D Goal Pose (default: policies/martha.npz)
#   ./robot.sh teleop             (another terminal) Xbox controller if plugged in, else keyboard
#   ./robot.sh check              bring-up tests of the ESP32 over serial, with nothing else running
#   ./robot.sh shell | build | stop
#
# teleop and policy both drive /cmd_vel: use one at a time. SPEED=0.5 ./robot.sh policy commands
# half the trained speeds. Other ports: ESP32=/dev/ttyUSB0 LIDAR=/dev/ttyUSB1 ./robot.sh mapping
set -e
cd "$(dirname "$0")"
source docker/lib.sh

ESP32=${ESP32:-/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0}
LIDAR=${LIDAR:-/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_9d653b21a3e51e42bf6b7027b0d61dd1-if00-port0}

# real.launch.py with launch arguments $1. A terminal closed instead of Ctrl-C leaves its nodes
# running in the container, and two bridges would then fight over the port: stop those first.
base() {
  for port in "$ESP32" "$LIDAR"; do
    [ -e "$port" ] || { say "$port is not plugged in (./robot.sh help for other ports)"; exit 1; }
  done
  up
  run 'pkill -f "[r]eal.launch.py"; pkill -x esp32_bridge; pkill -x rplidar_node; true' < /dev/null
  run "ros2 launch martha_nav real.launch.py esp32_port:=$ESP32 lidar_port:=$LIDAR $1"
}

case "$1" in
  mapping) base ;;
  save-map)
    [ -n "$2" ] || { help; exit 1; }
    up
    run "ros2 run nav2_map_server map_saver_cli -f $CT_REPO/maps/$2 -t /slam_map && \
      ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
      \"{filename: '$CT_REPO/maps/$2'}\""
    say "saved maps/$2: ./robot.sh $2 navigates in it" ;;
  policy) up; policy "$2" "speed_scale:=${SPEED:-1.0}" ;;
  teleop)
    up
    if compgen -G '/dev/input/js*' > /dev/null; then
      run 'trap "kill 0" EXIT
        ros2 run joy joy_node --ros-args --params-file config/xbox_teleop.yaml &
        ros2 run martha_nav xbox_teleop --ros-args --params-file config/xbox_teleop.yaml &
        wait'
    else
      run 'ros2 run teleop_twist_keyboard teleop_twist_keyboard'
    fi ;;
  check) python3 tools/bringup_esp32.py --port "$ESP32" "${@:2}" ;;
  shell|build|stop) "$1" ;;
  *)
    [ -n "$1" ] && [ -f "maps/$1.posegraph" ] || {
      help
      say "maps: $(ls maps/*.posegraph 2> /dev/null | xargs -rn1 basename | sed 's/.posegraph$//' | xargs)"
      exit 1
    }
    base "map:=$CT_REPO/maps/$1" ;;
esac
