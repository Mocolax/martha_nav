#!/usr/bin/env bash
# Demo: play the same policy in several Gazebo worlds, a few episodes each.
# Runs inside the ros2_humble container, from the root of the repository.
#
#   ./tools/run_demo_worlds.sh                       # 10 episodes per world, headless
#   EPISODES=5 GUI=true WORLDS=lab ./tools/run_demo_worlds.sh
#
# For each world it starts sim.launch.py, waits for the policy, runs evaluate_gazebo on
# the reserved seeds, tears the simulation down and moves on. The episodes are the
# ones the 2D simulator plays for those seeds, so the numbers are comparable.
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
source /home/ros/ros2_ws/install/setup.bash

MODEL=${MODEL:-runs/wide_dyn_s0/best_model.zip}
EPISODES=${EPISODES:-10}
GUI=${GUI:-false}
RVIZ=${RVIZ:-$GUI}
SPEED=${SPEED:-1.0}
CONDITION=${CONDITION:-obstacles}
WORLDS=${WORLDS:-"room hall four_rooms multi tube roblab lab"}
OUT=${OUT:-runs/demo_worlds}

say () { echo "[$(date +%H:%M:%S)] $*"; }
[ -f "$MODEL" ] || { say "no existe $MODEL"; exit 1; }
mkdir -p "$OUT"
# Absolutas, para aceptar tanto rutas relativas al repo como completas.
MODEL=$(realpath "$MODEL")
OUT=$(realpath "$OUT")

stop_sim () {
  [ -n "${LAUNCH_PID:-}" ] && kill -INT -- "-$LAUNCH_PID" 2>/dev/null
  wait "${LAUNCH_PID:-0}" 2>/dev/null
  pkill -f '[g]zserver'; pkill -f '[g]zclient'
  sleep 3
}

run_world () {
  local world=$1 log="$OUT/launch_$1.log" xy waited=0
  # Spawn where the first episode starts, so the robot never appears inside a wall.
  xy=$(python3 -c "
from martha_nav.sim2d.env import eval_seeds
from martha_nav.sim2d.scenarios import ScenarioConfig, generate
s = generate(eval_seeds(1)[0], ScenarioConfig(sources=('$world',), obstacle_mode='always'))
print(f'{s.start[0]:.3f} {s.start[1]:.3f}')" 2>/dev/null | tail -1)
  [ -n "$xy" ] || { say "$world: no pude calcular el punto de aparición"; return 1; }

  say "$world: arrancando Gazebo en ($xy)"
  setsid ros2 launch martha_nav sim.launch.py world:="$world" gui:="$GUI" rviz:="$RVIZ" \
    sim_speed_factor:="$SPEED" checkpoint:="$MODEL" x:="${xy% *}" y:="${xy#* }" \
    > "$log" 2>&1 &
  LAUNCH_PID=$!

  until grep -q 'policy loaded' "$log" 2>/dev/null; do
    grep -q 'ppo_local_planner.*process has died' "$log" 2>/dev/null &&
      { say "$world: la política no cargó, ver $log"; stop_sim; return 1; }
    sleep 2
    waited=$((waited + 2))
    [ "$waited" -lt 120 ] || { say "$world: no quedó lista, ver $log"; stop_sim; return 1; }
  done
  sleep 5                                  # let the first scan and the map arrive

  say "$world: $EPISODES episodios"
  ros2 run martha_nav evaluate_gazebo --ros-args -p world:="$world" -p mode:=seeds \
    -p condition:="$CONDITION" -p episodes:="$EPISODES" -p out:="$OUT/demo_$world.csv" \
    2>&1 | grep -E 'seed |done:'
  stop_sim
}

trap 'stop_sim; exit 130' INT TERM
say "=== demo con $MODEL, $EPISODES episodios por mundo"
for world in $WORLDS; do
  run_world "$world"
done

ls "$OUT"/demo_*.csv > /dev/null 2>&1 || { say "sin resultados, revisa $OUT/launch_*.log"; exit 1; }
python3 - "$OUT" <<'PY'
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
    lines.append(f'| `{csv.stem[5:]}` | {len(d)} | ' + ' | '.join(f'{r:.0%}' for r in rates)
                 + (' | — |' if pd.isna(secs) else f' | {secs:.0f} |'))
    total, hits = total + len(d), hits + int((d.outcome == 'success').sum())
lines.append(f'| **total** | {total} | **{hits / total:.0%}** | | | | |')
text = '\n'.join(lines)
(out / 'resumen.md').write_text(text + '\n')
print(text)
PY
say "=== listo, CSV y resumen en $OUT"
