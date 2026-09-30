#!/usr/bin/env bash
COMMANDS=(
  "python deploy/joystick/joystick_ros.py "
  "python deploy/logger/log.py --mode hw "
  "python deploy/hardware/control_29dof_crawl_diff_drive.py --config g1_29dof_crawl_diff_drive "
  "python deploy/hardware/hardware.py --config g1_29dof_crawl_diff_drive --network enp86s0 "
)

source "$(dirname "${BASH_SOURCE[0]}")/deploy.sh"
