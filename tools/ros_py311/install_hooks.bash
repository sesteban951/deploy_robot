#!/usr/bin/env bash
# Install only our hooks and ros2 wrapper in the active deployment environment.
set -eo pipefail
kcf_ros_tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kcf_ros_deploy="$(cd -- "$kcf_ros_tools/../.." && pwd)"
if [[ -z "${CONDA_PREFIX:-}" || ! -f "$CONDA_PREFIX/conda-meta/history" ]]; then
    printf '%s\n' 'Run this through make kcf-setup in the deployment Conda environment.' >&2
    exit 1
fi
"$CONDA_PREFIX/bin/python" -c 'import sys,numpy; assert sys.version_info[:2] == (3,11) and int(numpy.__version__.split(".")[0]) == 2'
for kcf_ros_file in "$CONDA_PREFIX/etc/conda/activate.d/zz_kcf_ros.sh" \
                    "$CONDA_PREFIX/etc/conda/deactivate.d/zz_kcf_ros.sh" \
                    "$CONDA_PREFIX/bin/ros2"; do
    if [[ -e "$kcf_ros_file" ]]; then
        kcf_ros_header="$(head -n 2 "$kcf_ros_file")"
        if [[ "$kcf_ros_header" != *'KCF ROS'* && "$kcf_ros_header" != *'matching Conda Python'* \
              && "$kcf_ros_header" != *'Restore only the ROS-related exports changed by our activation hook.'* ]]; then
            printf 'Refusing to overwrite an unrelated file: %s\n' "$kcf_ros_file" >&2
            exit 1
        fi
    fi
done
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
install -m 644 "$kcf_ros_tools/conda_activate.sh" "$CONDA_PREFIX/etc/conda/activate.d/zz_kcf_ros.sh"
install -m 644 "$kcf_ros_tools/conda_deactivate.sh" "$CONDA_PREFIX/etc/conda/deactivate.d/zz_kcf_ros.sh"
install -m 755 "$kcf_ros_tools/ros2_py311.sh" "$CONDA_PREFIX/bin/ros2"
# Scope the checkout path to this environment, never to ~/.bashrc.
"${CONDA_EXE:-conda}" env config vars set -p "$CONDA_PREFIX" DEPLOY_ROOT_DIR="$kcf_ros_deploy"
printf '%s\n' 'KCF ROS installed. Use conda activate deploy; no extra source command.'
