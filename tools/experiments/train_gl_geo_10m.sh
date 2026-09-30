#!/usr/bin/env bash
# Arm GL geodesic, 10M steps: LSTM, holonomic (vx, vy, w), goal only, geodesic progress.
# The 5M run was still improving (std 0.39, training success 0.62 -> 0.69 -> 0.76 over the
# last 1.5M), so it is retrained from scratch with twice the budget: the learning rate decays
# linearly to zero at the end, so the 5M run cannot simply be continued.
#
# Runs inside the ros2_humble container, from the root of the repository:
#   ./tools/experiments/train_gl_geo_10m.sh
#
# Variables: PY (python), NAME, STEPS, DEVICE (auto | cuda | cpu).
set -e
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
NAME=${NAME:-armGLgeo10M_s0}
STEPS=${STEPS:-10000000}
DEVICE=${DEVICE:-auto}
say () { echo "[$(date +%H:%M)] $*"; }

if [ ! -f "runs/$NAME/last_model.zip" ]; then
  rm -rf "runs/$NAME"          # an interrupted run cannot be resumed; start over
  say "entrenando $NAME ($STEPS pasos)"
  $PY -m martha_nav.learning.train --preset full --arch cnn --seed 0 --name "$NAME" \
    --steps "$STEPS" --device "$DEVICE" --recurrent \
    --action-dim 3 --target goal --reward-progress geodesic
fi

m="runs/$NAME/best_model.zip"
say "evaluando $m"
$PY -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition clean
$PY -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition obstacles
$PY -m martha_nav.learning.evaluate --model "$m" --episodes 200 --condition obstacles --sources lab
$PY -m martha_nav.learning.evaluate --model "$m" --condition obstacles --points lab
$PY tools/plot_report.py "runs/$NAME" > /dev/null
say "listo: copia la carpeta runs/$NAME a la máquina principal"
