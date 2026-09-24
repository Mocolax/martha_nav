#!/usr/bin/env bash
# Stuck-signal ablation: 2x2 against the reference.
#
#   sin señal / sin LSTM -> long_c_kl_s0 (referencia, ya entrenada)
#   sin señal / con LSTM -> armL_lstm_s0 (ya entrenada)
#   con señal / sin LSTM -> armS_stuck_s0
#   con señal / con LSTM -> armSL_s0
#
# Waits for arm S to finish, evaluates it, then trains and evaluates arm SL.
cd "$(dirname "$0")/.."
exec > >(tee -a runs/arms_stuck.log) 2>&1
say () { echo "[$(date +%H:%M)] $*"; }

evaluate () {
  local m="runs/$1/best_model.zip"
  [ -f "$m" ] || { say "AVISO: $1 sin best_model.zip"; return 1; }
  say "evaluando $1"
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition clean 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition obstacles 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 200 --condition obstacles --sources lab 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --condition obstacles --points lab 2>&1 | grep -v -i warn
  ./tools/ct python3 tools/plot_report.py "runs/$1" > /dev/null && say "informes de $1 listos"
}

say "esperando a que termine armS_stuck_s0"
until [ -f runs/armS_stuck_s0/last_model.zip ]; do
  docker exec ros2_humble pgrep -f "[l]earning.train" > /dev/null || sleep 60
  sleep 60
done
say "armS_stuck_s0 terminado"
evaluate armS_stuck_s0

if [ ! -f runs/armSL_s0/last_model.zip ]; then
  say "entrenando armSL_s0 (LSTM + señal de atasco)"
  rm -rf runs/armSL_s0
  ./tools/ct python3 -m martha_nav.learning.train --preset full --arch cnn --seed 0 \
    --name armSL_s0 --stuck-signal --recurrent 2>&1 | grep -viE "warn" | tail -3
fi
evaluate armSL_s0

say "tabla 2x2"
./tools/ct python3 - <<'PY' | tee -a docs/resultados-atasco.md
from pathlib import Path

import pandas as pd

from martha_nav.learning.evaluate import summarize

RUNS = [('sin señal, sin LSTM (referencia)', 'long_c_kl_s0'),
        ('sin señal, con LSTM (brazo L)', 'armL_lstm_s0'),
        ('con señal, sin LSTM (brazo S)', 'armS_stuck_s0'),
        ('con señal, con LSTM (brazo SL)', 'armSL_s0')]
FILES = [('limpio', 'eval_clean_train.csv'), ('obstáculos', 'eval_obstacles_train.csv'),
         ('lab', 'eval_obstacles_lab.csv'), ('lab, puntos fijos', 'eval_obstacles_lab-points.csv')]
print('\n| run | ' + ' | '.join(n for n, _ in FILES) + ' | estancado (obst.) |')
print('|---|' + '---|' * (len(FILES) + 1))
for label, run in RUNS:
    cells, stalled = [], '—'
    for name, fname in FILES:
        path = Path('runs') / run / fname
        if not path.exists():
            cells.append('—')
            continue
        s = summarize(pd.read_csv(path).to_dict('records'))
        cells.append(f"{s['success']:.3f} [{s['success_ci'][0]:.3f}, {s['success_ci'][1]:.3f}]")
        if name == 'obstáculos':
            stalled = f"{s['stalled']:.3f}"
    print(f'| {label} | ' + ' | '.join(cells) + f' | {stalled} |')
PY
say "=== ablación de la señal de atasco terminada"
