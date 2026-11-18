#!/usr/bin/env bash
# Installed as deploy/bin/ros2, beside the matching Conda Python interpreter.
exec "$(dirname -- "$0")/python" /opt/ros/humble/bin/ros2 "$@"
