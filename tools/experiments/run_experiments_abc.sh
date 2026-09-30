#!/usr/bin/env bash
# Experiments A/B/C (docs/resultados.md, "Primer run completo"): 2M steps each, then
# deterministic evaluation of each best model with the same seeds as full_cnn_s0.
set -e
cd "$(dirname "$0")/../.."
declare -A FLAGS=(
  [expA_col20]="--reward-collision -20"
  [expB_col20_stall5]="--reward-collision -20 --reward-stalled -5"
  [expC_col20_inverse]="--reward-collision -20 --lidar-encoding inverse"
)
for name in expA_col20 expB_col20_stall5 expC_col20_inverse; do
  echo "=== $name: ${FLAGS[$name]}"
  ./tools/ct python3 -m martha_nav.learning.train --preset full --arch cnn --seed 0 --steps 2000000 \
    --name "$name" ${FLAGS[$name]} 2>&1 | grep -v -i warn
  m="runs/$name/best_model.zip"
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition clean 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition obstacles 2>&1 | grep -v -i warn
  ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 200 --condition obstacles --sources lab 2>&1 | grep -v -i warn
  ./tools/ct python3 tools/experiments/plot_run.py "runs/$name"
done
echo "=== all done"
