#!/usr/bin/env bash
# Copy martha_nav and an exported policy to the Burger's Raspberry Pi and build it there.
#   ./tools/deploy_burger.sh ubuntu@192.168.1.50 runs/burger_s0/policy.npz
# On the Pi, once:
#   sudo apt install ros-humble-slam-toolbox ros-humble-nav2-map-server ros-humble-teleop-twist-keyboard python3-scipy
set -e
cd "$(dirname "$0")/.."
usage="usage: $0 user@host policy.npz"
host=${1:?$usage}
policy=${2:?$usage}
[[ $policy == *.npz && -f $policy ]] || { echo "$policy is not an existing .npz (export_policy writes it)" >&2; exit 1; }
ws='~/martha_ws'
ssh "$host" "mkdir -p $ws/src/martha_nav/policies"
rsync -a --delete --exclude runs/ --exclude maps/ --exclude policies/ --exclude 'entrega_tesis*' \
  --exclude graphify-out/ --exclude __pycache__/ --exclude .pytest_cache/ --exclude .superpowers/ \
  --exclude .git/ ./ "$host:$ws/src/martha_nav/"
rsync -a "$policy" "$host:$ws/src/martha_nav/policies/"
ssh "$host" "source /opt/ros/humble/setup.bash && cd $ws && colcon build --symlink-install --packages-select martha_nav"
echo "en el robot: source $ws/install/setup.bash && ros2 launch martha_nav burger.launch.py map:=... checkpoint:=\$HOME${ws#\~}/src/martha_nav/policies/$(basename "$policy")"
