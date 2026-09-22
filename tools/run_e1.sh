#!/usr/bin/env bash
# E1 (spec section 9): CNN vs MLP, 3 seeds each, 5M steps, current defaults
# (LiDAR d/(d+1), collision -20, target_kl 0.02). Each best model is then evaluated
# on the reserved seeds: clean, with obstacles, and in the unseen lab.
set -e
cd "$(dirname "$0")/.."
for seed in 0 1 2; do
  for arch in cnn mlp; do
    name="e1_${arch}_s${seed}"
    [ -f "runs/$name/last_model.zip" ] && { echo "=== $name already done"; continue; }
    echo "=== $name"
    rm -rf "runs/$name"
    ./tools/ct python3 -m martha_nav.learning.train --preset full --arch "$arch" --seed "$seed" \
      --eval-every 250000 --name "$name" 2>&1 | grep -v -i warn | tail -4
    m="runs/$name/best_model.zip"
    ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition clean 2>&1 | grep -v -i warn
    ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 500 --condition obstacles 2>&1 | grep -v -i warn
    ./tools/ct python3 -m martha_nav.learning.evaluate --model "$m" --episodes 200 --condition obstacles --sources lab 2>&1 | grep -v -i warn
  done
done
echo "=== E1 done"
