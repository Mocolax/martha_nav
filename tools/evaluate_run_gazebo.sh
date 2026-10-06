#!/usr/bin/env bash
# The standard Gazebo evaluation of a run's best model, as evaluate_run.sh is for 2D:
# launches N Gazebos side by side (each with its own ROS domain and Gazebo port) and runs the
# evaluate_gazebo node in each, with the physics unthrottled (~3x real time each; checked
# against 1x in docs/resultados.md).
# shard:=i/N splits the seeds between them; the parts are merged at the end.
#   [EPISODES=100] ./tools/evaluate_run_gazebo.sh runs/wide_dyn_s0 "" [instances] [out_dir]
# -> <out_dir, default the run>/eval_gazebo_lab<suffix>.csv, eval_gazebo_lab_points<suffix>.csv
#    (+ _traj/_route/_obstacles beside each)
set -e
cd "$(dirname "$0")/.."
usage="usage: $0 runs/<name> <suffix> [instances] [out_dir]"
run=${1:?$usage}
suffix=${2?$usage}
n=${3:-3}
out=${4:-$run}
mkdir -p "$out"
ws=/home/ros/ros2_ws/src/martha_nav
ros='source /opt/ros/humble/setup.bash; source /home/ros/ros2_ws/install/setup.bash'

stop() {
  for i in $(seq 0 $((n - 1))); do
    docker exec ros2_humble bash -c "kill -INT -- -\$(cat /tmp/sim$i.pgid) 2>/dev/null" || true
  done
}
trap stop EXIT

for i in $(seq 0 $((n - 1))); do
  docker exec ros2_humble rm -f /tmp/sim$i.log /tmp/sim$i.pgid
  docker exec -d ros2_humble bash -c "$ros; export ROS_DOMAIN_ID=$((60 + i)) \
    GAZEBO_MASTER_URI=http://localhost:$((11360 + i)); setsid bash -c 'echo \$\$ > /tmp/sim$i.pgid; \
    exec ros2 launch martha_nav sim.launch.py world:=lab gui:=false x:=0.95 y:=1.35 \
    checkpoint:=$ws/$run/best_model.zip' > /tmp/sim$i.log 2>&1"
done
for i in $(seq 0 $((n - 1))); do
  timeout 180 bash -c "until docker exec ros2_humble grep -qs 'action space' /tmp/sim$i.log; do sleep 3; done"
  docker exec ros2_humble bash -c "$ros; GAZEBO_MASTER_URI=http://localhost:$((11360 + i)) gz physics -u 0"
done

for mode in seeds points; do
  name=eval_gazebo_lab$suffix
  if [ $mode = points ]; then name=eval_gazebo_lab_points$suffix; fi
  for i in $(seq 0 $((n - 1))); do
    ./tools/ct_ros env ROS_DOMAIN_ID=$((60 + i)) ros2 run martha_nav evaluate_gazebo --ros-args \
      -p mode:=$mode -p episodes:=${EPISODES:-100} -p condition:=obstacles -p shard:=$i/$n \
      -p out:=$ws/$out/$name.part$i.csv </dev/null > "$out/$name.part$i.log" 2>&1 &
  done
  wait
  ./tools/ct python3 - "$out/$name" "$n" <<'EOF'
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
