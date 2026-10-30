#!/usr/bin/env bash
# Gazebo step test of the mecanum drive (tools/step_response.py) on martha_tower, in room.world's
# most open spot, without the policy. -> runs/step_response.csv
set -e
cd "$(dirname "$0")/.."
ros='source /opt/ros/humble/setup.bash; source /home/ros/ros2_ws/install/setup.bash'
stop() { docker exec ros2_humble bash -c 'kill -INT -- -$(cat /tmp/step.pgid) 2>/dev/null' || true; }
trap stop EXIT
docker exec ros2_humble rm -f /tmp/step.log /tmp/step.pgid
docker exec -d ros2_humble bash -c "$ros; export ROS_DOMAIN_ID=70 GAZEBO_MASTER_URI=http://localhost:11370; \
  setsid bash -c 'echo \$\$ > /tmp/step.pgid; exec ros2 launch martha_nav sim.launch.py world:=room \
  robot:=martha_tower gui:=false x:=-6.4 y:=-1.4' > /tmp/step.log 2>&1"
timeout 180 bash -c "until docker exec ros2_humble grep -qsiE 'activated.*mecanum' /tmp/step.log; do sleep 3; done"
sleep 3
./tools/ct bash -c "$ros; export ROS_DOMAIN_ID=70; python3 tools/step_response.py runs/step_response.csv"
