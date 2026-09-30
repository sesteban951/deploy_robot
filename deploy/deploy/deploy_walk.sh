#!/usr/bin/env bash
COMMANDS=(
  "python deploy/joystick/joystick_ros.py "
  "python deploy/logger/log.py --mode hw "
  "python deploy/hardware/control_29dof_walk_unicycle_lstm.py --config g1_29dof_walk_unicycle_lstm "
  "python deploy/hardware/hardware.py --config g1_29dof_walk_unicycle_lstm --network enp86s0 "
)

source "$(dirname "${BASH_SOURCE[0]}")/deploy.sh"
