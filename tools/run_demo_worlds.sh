#!/usr/bin/env bash
# Demo: play the same policy in every Gazebo world, a few episodes each.
#
#   ./tools/run_demo_worlds.sh                                   # 10 episodes per world, headless
#   EPISODES=5 GUI=true WORLDS="lab room" ./tools/run_demo_worlds.sh
#
# For each world it starts sim.launch.py, waits for the controller and the policy,
# runs gazebo_eval on the reserved seeds, tears the simulation down and moves on.
# Every episode is the same one the 2D simulator plays for that seed, so the numbers
# are comparable with docs/resultados.md.
set -u
cd "$(dirname "$0")/.."

MODEL=${MODEL:-runs/long_c_kl_s0/best_model.zip}
EPISODES=${EPISODES:-10}
GUI=${GUI:-false}
RVIZ=${RVIZ:-$GUI}
SPEED=${SPEED:-1.0}
CONDITION=${CONDITION:-obstacles}
WORLDS=${WORLDS:-"room hall four_rooms multi tube roblab lab"}
OUT=${OUT:-runs/demo_worlds}
READY_TIMEOUT=${READY_TIMEOUT:-120}

IN_CONTAINER=/home/ros/ros2_ws/src/martha_nav
mkdir -p "$OUT"
say () { echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$MODEL" ] || { say "no existe $MODEL"; exit 1; }
# MODEL y OUT viajan al contenedor como rutas relativas al repo.
for path in "$MODEL" "$OUT"; do
  case "$path" in /*) say "ruta absoluta: $path; MODEL y OUT son relativos al repo"; exit 1;; esac
done

stop_sim () {
  kill "${LAUNCH_PID:-0}" 2>/dev/null
  for p in '[r]os2 launch martha_nav' '[g]zclient' '[g]zserver' '[p]olicy_node' \
           '[p]lanner_node' '[m]ap_publisher' '[g]round_truth_tf' '[c]md_vel_bridge' \
           '[r]obot_state_publisher' '[r]viz2'; do
    docker exec ros2_humble pkill -f "$p" 2>/dev/null
  done
  wait "${LAUNCH_PID:-0}" 2>/dev/null
  sleep 3
}

wait_until_ready () {   # $1 = launch log
  local waited=0
  while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if grep -q 'policy loaded' "$1" 2>/dev/null &&
       grep -qE 'mecanum_drive_controller|planar' "$1" 2>/dev/null; then
      sleep 5                       # let the first scan and the map arrive
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

run_world () {
  local world=$1
  local log="$OUT/launch_$world.log"
  # Spawn where the first episode starts, so the robot never appears inside a wall.
  local xy
  xy=$(./tools/ct python3 -c "
from martha_nav.learning.evaluate import eval_seeds
from martha_nav.sim2d.scenarios import ScenarioConfig, generate
s = generate(eval_seeds(1)[0], ScenarioConfig(sources=('$world',), obstacle_mode='always'))
print(f'{s.start[0]:.3f} {s.start[1]:.3f}')" 2>/dev/null | tail -1)
  [ -n "$xy" ] || { say "$world: no pude calcular el punto de aparición"; return 1; }

  say "$world: arrancando Gazebo en ($xy)"
  ./tools/ct_ros bash -c "source /home/ros/ros2_ws/install/setup.bash && exec ros2 launch martha_nav sim.launch.py \
      world:=$world gui:=$GUI rviz:=$RVIZ sim_speed_factor:=$SPEED \
      checkpoint:=$IN_CONTAINER/$MODEL x:=${xy% *} y:=${xy#* }" > "$log" 2>&1 &
  LAUNCH_PID=$!

  if ! wait_until_ready "$log"; then
    say "$world: la simulación no quedó lista, ver $log"
    stop_sim
    return 1
  fi

  say "$world: $EPISODES episodios"
  ./tools/ct_ros bash -c "source /home/ros/ros2_ws/install/setup.bash && \
      ros2 run martha_nav gazebo_eval --ros-args -p world:=$world -p mode:=seeds \
      -p condition:=$CONDITION -p episodes:=$EPISODES \
      -p out:=$IN_CONTAINER/$OUT/demo_$world.csv" 2>&1 | grep -E 'seed |done:'
  stop_sim
}

summarise () {
  ./tools/ct python3 - "$OUT" <<'PY'
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
lines = ['| mundo | episodios | éxito | colisión | estancado | timeout | segundos (éxitos) |',
         '|---|---|---|---|---|---|---|']
total = hits = 0
for csv in sorted(out.glob('demo_*.csv')):
    d = pd.read_csv(csv)
    rates = [(d.outcome == o).mean() for o in ('success', 'collision', 'stalled', 'timeout')]
    secs = d.seconds[d.outcome == 'success'].mean()
    cell = '—' if pd.isna(secs) else f'{secs:.0f}'
    lines.append(f"| `{csv.stem[5:]}` | {len(d)} | "
                 + ' | '.join(f'{r:.0%}' for r in rates) + f' | {cell} |')
    total += len(d)
    hits += int((d.outcome == 'success').sum())
if total:
    lines.append(f'| **total** | {total} | **{hits / total:.0%}** | | | | |')
text = '\n'.join(lines)
(out / 'resumen.md').write_text(text + '\n')
print(text)
PY
}

trap 'stop_sim; exit 130' INT TERM
say "=== demo con $MODEL, $EPISODES episodios por mundo"
for world in $WORLDS; do
  run_world "$world"
done
say "=== resumen"
if ls "$OUT"/demo_*.csv > /dev/null 2>&1; then
  summarise
else
  say "ningún mundo dejó resultados, revisa $OUT/launch_*.log"
fi
say "=== listo, CSV y resumen en $OUT"
