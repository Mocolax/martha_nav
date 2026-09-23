#!/usr/bin/env bash
# E2: the same episodes in Gazebo as in the 2D simulator (same seeds, paired).
# Needs sim.launch.py already running with the checkpoint.
set -e
cd "$(dirname "$0")/.."
R=runs/long_c_kl_s0
./tools/ct_ros bash -c "source /home/ros/ros2_ws/install/setup.bash && ros2 run martha_nav gazebo_eval --ros-args -p episodes:=100 -p mode:=seeds -p condition:=obstacles -p out:=/home/ros/ros2_ws/src/martha_nav/$R/eval_gazebo_lab.csv" 2>&1 | grep -E "done:" 
./tools/ct_ros bash -c "source /home/ros/ros2_ws/install/setup.bash && ros2 run martha_nav gazebo_eval --ros-args -p mode:=points -p condition:=obstacles -p out:=/home/ros/ros2_ws/src/martha_nav/$R/eval_gazebo_lab_points.csv" 2>&1 | grep -E "done:"
echo "=== E2 done"
