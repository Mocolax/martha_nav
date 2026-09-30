#!/usr/bin/env bash
# The standard 2D evaluation of a run's best model, then its report:
# clean and with obstacles (500 episodes each), the unseen lab (200) and its fixed points.
#   ./tools/evaluate_run.sh runs/mi_run
set -e
cd "$(dirname "$0")/.."
run=${1:?usage: $0 runs/<name>}
model="$run/best_model.zip"
./tools/ct_ros ros2 run martha_nav evaluate_2d --model "$model" --episodes 500 --condition clean
./tools/ct_ros ros2 run martha_nav evaluate_2d --model "$model" --episodes 500 --condition obstacles
./tools/ct_ros ros2 run martha_nav evaluate_2d --model "$model" --episodes 200 --condition obstacles --sources lab
./tools/ct_ros ros2 run martha_nav evaluate_2d --model "$model" --condition obstacles --points lab
./tools/ct python3 tools/plot_report.py "$run"
