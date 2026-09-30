#!/usr/bin/env bash
# Overnight queue: evaluate arm H when it finishes, then train and evaluate arm L
# (LSTM) and arm HL (both), writing a summary to docs/resultados-noche.md.
#
# Every step is independent: a failure is logged and the queue moves on.
cd "$(dirname "$0")/../.."
LOG=runs/night.log
SUMMARY=docs/resultados-noche.md
exec > >(tee -a "$LOG") 2>&1

say () { echo "[$(date +%H:%M)] $*"; }

wait_for_training () {   # $1 = run name
  say "esperando a que termine $1"
  while [ ! -f "runs/$1/last_model.zip" ]; do
    if ! docker exec ros2_humble pgrep -f "[l]earning.train" > /dev/null; then
      sleep 30                                   # give the process time to write the file
      [ -f "runs/$1/last_model.zip" ] || { say "AVISO: $1 no dejó last_model.zip"; return 1; }
    fi
    sleep 60
  done
  say "$1 terminado"
}

evaluate () {            # $1 = run name
  local m="runs/$1/best_model.zip"
  [ -f "$m" ] || { say "AVISO: $1 sin best_model.zip, no se evalúa"; return 1; }
  say "evaluando $1"
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition clean 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition obstacles 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 200 --condition obstacles --sources lab 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --condition obstacles --points lab 2>&1 | grep -v -i warn
  ./tools/ct python3 tools/plot_report.py "runs/$1" > /dev/null && say "informes de $1 listos"
}

train () {               # $1 = run name, rest = flags
  local name=$1; shift
  [ -f "runs/$name/last_model.zip" ] && { say "$name ya estaba entrenado"; return 0; }
  say "entrenando $name ($*)"
  rm -rf "runs/$name"
  ./tools/ct python3 -m martha_nav.learning.train --preset full --arch cnn --seed 0 \
    --name "$name" "$@" 2>&1 | grep -viE "warn" | tail -3
}

summarise () {
  say "escribiendo $SUMMARY"
  ./tools/ct python3 - <<'PY' >> "$SUMMARY"
from pathlib import Path
import pandas as pd
from martha_nav.learning.evaluate import summarize
RUNS = [('referencia (v, w)', 'long_c_kl_s0'), ('brazo H (vx, vy, w)', 'armH_holonomic_s0'),
        ('brazo L (LSTM)', 'armL_lstm_s0'), ('brazo HL (ambos)', 'armHL_s0')]
FILES = [('limpio', 'eval_clean_train.csv'), ('obstáculos', 'eval_obstacles_train.csv'),
         ('lab', 'eval_obstacles_lab.csv'), ('lab, puntos fijos', 'eval_obstacles_lab-points.csv')]
print('\n| run | ' + ' | '.join(name for name, _ in FILES) + ' |')
print('|---|' + '---|' * len(FILES))
for label, run in RUNS:
    cells = []
    for _, fname in FILES:
        path = Path('runs') / run / fname
        if not path.exists():
            cells.append('—')
            continue
        s = summarize(pd.read_csv(path).to_dict('records'))
        cells.append(f"{s['success']:.3f} [{s['success_ci'][0]:.3f}, {s['success_ci'][1]:.3f}]")
    print(f'| {label} | ' + ' | '.join(cells) + ' |')
PY
}

say "=== cola nocturna arrancada"
mkdir -p docs
{ echo; echo "## Cola nocturna del $(date +%Y-%m-%d)"; echo;
  echo 'Brazos comparados contra la referencia `long_c_kl_s0`. Éxito con IC95, evaluación determinista.'; } >> "$SUMMARY"

wait_for_training armH_holonomic_s0 && evaluate armH_holonomic_s0
train armL_lstm_s0 --recurrent && evaluate armL_lstm_s0
train armHL_s0 --recurrent --action-dim 3 && evaluate armHL_s0
summarise
say "=== cola nocturna terminada"
