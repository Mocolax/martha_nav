#!/usr/bin/env bash
# Save the map slam_toolbox is building, for real.launch.py / burger.launch.py map:=<out>:
#   <out>.pgm/.yaml       the frozen /map of the global planner (editable in GIMP)
#   <out>.posegraph/.data what slam_toolbox localizes against
# Usage (with the launch mapping): tools/save_map.sh /abs/path/martha_nav/maps/lab_real
set -euo pipefail
out=${1:?usage: $0 /abs/path/without/extension}
mkdir -p "$(dirname "$out")"
ros2 run nav2_map_server map_saver_cli -f "$out" -t /slam_map
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$out'}"
