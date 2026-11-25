# What sim.sh and robot.sh share: the ros2_humble container (compose.yaml) and how to run
# things in it. Sourced from the repository's root.

CT=ros2_humble
CT_REPO=/home/ros/ros2_ws/src/martha_nav        # this repository, inside the container
DC="docker compose -f $PWD/docker/compose.yaml"

# compose.yaml's variables. The repository sits in a colcon workspace, $WS/src/martha_nav.
gid() { getent group "$1" | cut -d: -f3 | grep . || echo "$2"; }
export UID GID=$(id -g) WS=$(realpath ../..) DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR
export DIALOUT_GID=$(gid dialout 20) VIDEO_GID=$(gid video 44)
export RENDER_GID=$(gid render "$VIDEO_GID")

say() { echo "[martha] $*"; }

# The comment block at the top of the calling script, without its "#".
help() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

# A shell command line in the container, in the repository, with ROS and the workspace sourced.
# With a terminal it gets one too, so Ctrl-C reaches the command.
run() {
  docker exec -i $([ -t 0 ] && [ -t 1 ] && echo -t) -w $CT_REPO $CT bash -c '
    source /opt/ros/humble/setup.bash
    [ -f /home/ros/ros2_ws/install/setup.bash ] && source /home/ros/ros2_ws/install/setup.bash
    eval "$1"' _ "$1"
}

# --symlink-install links each file; the links of files deleted since would break the build.
build() {
  run 'cd /home/ros/ros2_ws && find build/martha_nav install/martha_nav -xtype l -delete 2> /dev/null
    colcon build --symlink-install --packages-up-to martha_nav'
}
shell() { up; docker exec -it -w $CT_REPO $CT bash; }
stop() { $DC stop; }

# Start the container (building the image the first time) and build the workspace if needed.
up() {
  [ "$(docker inspect -f '{{.State.Running}}' $CT 2>/dev/null)" = true ] || $DC up -d
  xhost +local:docker > /dev/null 2>&1 || true            # let the container open windows
  [ -d "$WS/install/martha_nav" ] || build
}

# policy.launch.py with the policy $1 (default policies/martha.npz) and launch arguments $2.
policy() {
  local npz=${1:-policies/martha.npz}
  [ -f "$npz" ] || { say "$npz does not exist"; exit 1; }
  npz=$(realpath --relative-to=. "$npz")
  run "ros2 launch martha_nav policy.launch.py checkpoint:=$CT_REPO/$npz $2"
}

# The robot a policy was trained for, martha or martha_tower.
policy_robot() {
  run "python3 -c \"from martha_nav.robots import checkpoint_robot as r; print(r('$1'))\"" < /dev/null
}
