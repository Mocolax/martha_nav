#!/usr/bin/env bash
# E2 for one run, fast: N Gazebos side by side (each with its own ROS domain and Gazebo port)
# with the physics unthrottled (~3x real time each; checked against 1x in docs/resultados.md).
# shard:=i/N splits the seeds between them; the parts are merged at the end.
#   ./tools/evaluate_gazebo_fast.sh runs/wide_dyn_s0 _v5 [instances] [out_dir]
# -> <out_dir, default the run>/eval_gazebo_lab_v5.csv, eval_gazebo_lab_points_v5.csv
#    (+ _traj/_route/_obstacles beside each)
set -e
cd "$(dirname "$0")/.."
run=${1:?usage: $0 runs/<name> <suffix> [instances]}
suffix=${2?usage: $0 runs/<name> <suffix> [instances]}
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
    rows = [r for p in parts for r in csv.DictReader(open(p))]
    keys = list(dict.fromkeys(k for p in parts for k in (csv.DictReader(open(p)).fieldnames or [])))
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
