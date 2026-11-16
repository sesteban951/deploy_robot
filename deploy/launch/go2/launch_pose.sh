#!/usr/bin/env bash
COMMANDS=(
  "python deploy/joystick/joystick_ros.py --source xinput_usb "
  "python deploy/logger/log.py --mode hw --robot go2 "
  "python deploy/hardware/go2/control_pose.py --config go2_pose "
  "python deploy/hardware/go2/hardware.py --config go2_pose --network eth0 "
)

source "$(dirname "${BASH_SOURCE[0]}")/launch.sh"
