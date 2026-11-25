#!/usr/bin/env bash
# Martha in simulation, on the PC. Everything runs in the ros2_humble container (docker/).
#
#   ./sim.sh [world]              Gazebo + Martha + RViz, no policy (default: lab; see worlds/)
#   ./sim.sh policy [npz]         (another terminal) planners + PPO, goals with 2D Goal Pose
#                                 (default: policies/martha.npz)
#   ./sim.sh teleop               (another terminal) keyboard -> /cmd_vel (Shift = sideways)
#   ./sim.sh eval runs/<name>     the standard 2D evaluation of its best model, and its report
#   ./sim.sh eval-gazebo runs/<name> [suffix] [instances] [out_dir]
#                                 the lab episodes in Gazebo, 3 instances in parallel
#   ./sim.sh test | shell | build | stop
#
# GUI=false hides Gazebo's window, ROBOT=martha drops the tower, DRIVE=planar moves the body
# without wheel dynamics, EPISODES=20 shortens eval-gazebo (default 100), GUARD=false runs it
# without the policy's safety guard.
set -e
cd "$(dirname "$0")"
source docker/lib.sh

# The standard 2D evaluation: clean and with obstacles (500 episodes each), the unseen lab (200)
# and its fixed points, then the run's report. -> <run>/eval_*.csv, learning_report.png
eval_2d() {
  local model=$1/best_model.zip
  for args in '--episodes 500 --condition clean' '--episodes 500 --condition obstacles' \
              '--episodes 200 --condition obstacles --sources lab' '--condition obstacles --points lab'; do
    run "ros2 run martha_nav evaluate_2d --model $model $args"
  done
  run "python3 tools/plot_report.py $1"
}

# The standard Gazebo evaluation of a run's best model: N Gazebos side by side, each with its own
# ROS domain and Gazebo port, the physics capped at 3x real time (checked against 1x in
# docs/resultados.md), the seeds split between them (shard:=i/N) and the parts merged at the end.
# -> <out_dir, default the run>/eval_gazebo_lab<suffix>.csv, eval_gazebo_lab_points<suffix>.csv
#    (+ _traj/_route/_obstacles beside each)
eval_gazebo() {
  local run_dir=$1 suffix=$2 n=${3:-3} out=${4:-$1} robot i mode name
  robot=$(policy_robot "$run_dir/best_model.zip")
  mkdir -p "$out"
  # Built now: the trap runs after this function's locals are gone.
  local stop=''
  for i in $(seq 0 $((n - 1))); do
    stop+="docker exec $CT bash -c 'kill -INT -- -\$(cat /tmp/sim$i.pgid) 2> /dev/null' || true; "
  done
  trap "$stop" EXIT
  for i in $(seq 0 $((n - 1))); do
    docker exec $CT rm -f /tmp/sim$i.log /tmp/sim$i.pgid
    docker exec -d -w $CT_REPO $CT bash -c "source /home/ros/ros2_ws/install/setup.bash
      export ROS_DOMAIN_ID=$((60 + i)) GAZEBO_MASTER_URI=http://localhost:$((11360 + i))
      setsid bash -c 'echo \$\$ > /tmp/sim$i.pgid; exec ros2 launch martha_nav sim.launch.py \
        world:=lab gui:=false x:=0.95 y:=1.35 checkpoint:=$CT_REPO/$run_dir/best_model.zip \
        guard:=${GUARD:-true}' \
        > /tmp/sim$i.log 2>&1"
  done
  for i in $(seq 0 $((n - 1))); do
    timeout 180 bash -c "until docker exec $CT grep -qsE 'action space|process has died' /tmp/sim$i.log; do sleep 3; done"
    if ! docker exec $CT grep -qs 'action space' /tmp/sim$i.log; then
      docker exec $CT grep -m1 -E 'Error|error' /tmp/sim$i.log || true
      say "Gazebo $i did not start, see /tmp/sim$i.log in the container"; exit 1
    fi
    run "GAZEBO_MASTER_URI=http://localhost:$((11360 + i)) gz physics -u 3000" < /dev/null
  done
  for mode in seeds points; do
    name=eval_gazebo_lab$suffix
    [ $mode = seeds ] || name=eval_gazebo_lab_points$suffix
    for i in $(seq 0 $((n - 1))); do
      run "ROS_DOMAIN_ID=$((60 + i)) ros2 run martha_nav evaluate_gazebo --ros-args \
        -p robot:=$robot -p mode:=$mode -p episodes:=${EPISODES:-100} -p condition:=obstacles \
        -p shard:=$i/$n -p out:=$CT_REPO/$out/$name.part$i.csv" \
        < /dev/null > "$out/$name.part$i.log" 2>&1 &
    done
    wait
    run "python3 - $out/$name $n" < /dev/null <<'EOF'
import csv, sys
from pathlib import Path
base, n = sys.argv[1], int(sys.argv[2])
for side in ('', '_traj', '_route', '_obstacles'):
    parts = [Path(f'{base}.part{i}{side}.csv') for i in range(n)]
    readers = [csv.DictReader(p.open()) for p in parts]
    rows = [r for reader in readers for r in reader]
    keys = list(dict.fromkeys(k for reader in readers for k in reader.fieldnames or []))
    rows.sort(key=lambda r: int(r['episode_seed']))          # stable: steps keep their order
    with open(f'{base}{side}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    for p in parts:
        p.unlink()
EOF
    grep -h "done:" "$out/$name".part*.log | cut -c1-160
  done
}

case "${1:-lab}" in
  policy)
    up
    # The planners follow the robot the policy was trained for: it has to be the one in Gazebo.
    want=$(policy_robot "${2:-policies/martha.npz}")
    has=martha
    run 'ros2 param get /robot_state_publisher robot_description' < /dev/null | grep -q tower_post \
      && has=martha_tower
    [ "$want" = "$has" ] || { say "the policy drives $want, Gazebo has $has (ROBOT=$want ./sim.sh)"; exit 1; }
    drive=planar
    run 'ros2 topic list --no-daemon' < /dev/null | grep -q /mecanum_drive_controller/odometry \
      && drive=mecanum
    policy "$2" "sim:=true drive:=$drive" ;;
  teleop) up; run 'ros2 run teleop_twist_keyboard teleop_twist_keyboard' ;;
  eval) [ -f "$2/best_model.zip" ] || { help; exit 1; }; up; eval_2d "$2" ;;
  eval-gazebo) [ -f "$2/best_model.zip" ] || { help; exit 1; }; up; eval_gazebo "${@:2}" ;;
  test) up; run 'python3 -m pytest test -q' ;;
  shell|build|stop) "$1" ;;
  *)
    world=${1:-lab}
    [ -f "worlds/$world.world" ] || { help; exit 1; }
    up
    # Spawn where the first evaluation episode starts, so Martha never appears inside a wall.
    xy=$(run "python3 -c \"
from martha_nav.sim2d.env import eval_seeds
from martha_nav.sim2d.scenarios import ScenarioConfig, generate
s = generate(eval_seeds(1)[0], ScenarioConfig(sources=('$world',), obstacle_mode='always'))
print(f'x:={s.start[0]:.3f} y:={s.start[1]:.3f}')\"" < /dev/null | tail -1)
    say "Gazebo with $world. In other terminals: ./sim.sh policy, ./sim.sh teleop"
    run "ros2 launch martha_nav sim.launch.py world:=$world $xy rviz:=true gui:=${GUI:-true} \
      drive:=${DRIVE:-mecanum} ${ROBOT:+robot:=$ROBOT}" ;;
esac
