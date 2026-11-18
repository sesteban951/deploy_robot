#!/usr/bin/env bash
# Standard version-matched ROS Humble Python 3.11 / NumPy 2 overlay.
# System ROS is never installed into or modified.
set -eo pipefail
kcf_ros_script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kcf_ros_checkout_root="$(cd -- "$kcf_ros_script_root/../.." && pwd)"
export KCF_ROS_WORKSPACE="${KCF_ROS_WORKSPACE:-$kcf_ros_checkout_root/.cache/ros_py311}"
kcf_ros_build_root="$KCF_ROS_WORKSPACE"
kcf_ros_python="$kcf_ros_build_root/env/bin/python"
kcf_ros_pybind="$kcf_ros_build_root/env/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
kcf_ros_include="$("$kcf_ros_python" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
kcf_ros_library="$("$kcf_ros_python" -c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR"), "libpython" + sysconfig.get_python_version() + ".so"))')"
test -f "$kcf_ros_include/Python.h"
test -f "$kcf_ros_library"
# Build only against system Humble plus this workspace, not an inherited overlay.
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH PKG_CONFIG_PATH
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1
export CMAKE_BUILD_PARALLEL_LEVEL=4
export MAKEFLAGS=-j4
export CCACHE_DIR="$kcf_ros_build_root/cache/ccache"
kcf_ros_cmake_args=(
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
    -DPYTHON_EXECUTABLE="$kcf_ros_python"
    -DPython_EXECUTABLE="$kcf_ros_python"
    -DPython3_EXECUTABLE="$kcf_ros_python"
    # Humble's legacy FindPythonExtra calls FindPythonLibs independently.
    # Pin headers AND library; interpreter/suffix alone is insufficient.
    -DPYTHON_INCLUDE_DIR:PATH="$kcf_ros_include"
    -DPYTHON_LIBRARY:FILEPATH="$kcf_ros_library"
    -DPython3_INCLUDE_DIR:PATH="$kcf_ros_include"
    -DPython3_LIBRARY:FILEPATH="$kcf_ros_library"
    -U_numpy_h
    -DPython_FIND_STRATEGY=LOCATION -DPython3_FIND_STRATEGY=LOCATION
    -DPYTHON_INSTALL_DIR=lib/python3.11/site-packages
    -DPYBIND11_FINDPYTHON=ON -Dpybind11_DIR="$kcf_ros_pybind"
)

kcf_ros_build_packages() {
    /usr/bin/colcon --log-base "$kcf_ros_build_root/log" build \
        --base-paths "$kcf_ros_build_root/src" \
        --build-base "$kcf_ros_build_root/build" \
        --install-base "$kcf_ros_build_root/install" --merge-install \
        --parallel-workers 2 --packages-select "$@" \
        --allow-overriding "$@" --event-handlers console_cohesion- \
        --cmake-args --no-warn-unused-cli "${kcf_ros_cmake_args[@]}"
}

case "${1:-all}" in
    generator)
        # Colcon includes the generator's interface dependency even with
        # BUILD_TESTING=OFF. Build it first on a genuinely empty workspace.
        kcf_ros_build_packages test_interface_files rosidl_generator_py
        ;;
    interfaces)
        source "$kcf_ros_build_root/install/local_setup.bash"
        kcf_ros_build_packages \
            test_interface_files builtin_interfaces unique_identifier_msgs action_msgs rcl_interfaces \
            lifecycle_msgs composition_interfaces rosgraph_msgs statistics_msgs \
            std_msgs std_srvs actionlib_msgs diagnostic_msgs geometry_msgs \
            nav_msgs sensor_msgs shape_msgs stereo_msgs trajectory_msgs \
            visualization_msgs test_msgs
        ;;
    rclpy)
        source "$kcf_ros_build_root/install/local_setup.bash"
        kcf_ros_build_packages rclpy
        ;;
    all)
        bash "$kcf_ros_script_root/build_ros.bash" generator
        bash "$kcf_ros_script_root/build_ros.bash" interfaces
        bash "$kcf_ros_script_root/build_ros.bash" rclpy
        ;;
    *)
        printf '%s\n' 'Usage: bash build_ros.bash [generator|interfaces|rclpy|all]' >&2
        exit 2
        ;;
esac
