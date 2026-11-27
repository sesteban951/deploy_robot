#!/usr/bin/env bash
COMMANDS=(
  "python deploy/joystick/joystick_ros.py "
  "python deploy/logger/log.py --mode hw "
  "python deploy/hardware/g1/control_29dof_walkjog_unicycle.py --config g1_29dof_walkjog_unicycle "
  "python deploy/hardware/g1/hardware.py --config g1_29dof_walkjog_unicycle --network enp86s0 "
)

source "$(dirname "${BASH_SOURCE[0]}")/launch.sh"
