#!/usr/bin/env bash
# The first E2 (long_c_kl_s0, unsuffixed CSVs): the same episodes in Gazebo as in 2D.
# Needs sim.launch.py already running with the checkpoint. Now: tools/evaluate_run_gazebo.sh.
set -e
cd "$(dirname "$0")/../.."
R=runs/long_c_kl_s0
./tools/ct_ros ros2 run martha_nav evaluate_gazebo --ros-args -p episodes:=100 -p mode:=seeds -p condition:=obstacles -p out:=/home/ros/ros2_ws/src/martha_nav/$R/eval_gazebo_lab.csv 2>&1 | grep -E "done:"
./tools/ct_ros ros2 run martha_nav evaluate_gazebo --ros-args -p mode:=points -p condition:=obstacles -p out:=/home/ros/ros2_ws/src/martha_nav/$R/eval_gazebo_lab_points.csv 2>&1 | grep -E "done:"
echo "=== E2 done"
