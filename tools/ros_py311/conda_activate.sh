# KCF ROS hook: installed by make install; sourced by conda activate deploy.
# Keep Conda's Python/PATH, without activating a nested virtual environment.
if [[ -z "${BASH_VERSION:-}" || -n "${_KCF_ROS_CONDA_ACTIVE:-}" ]]; then
    return 0
fi
if [[ -z "${DEPLOY_ROOT_DIR:-}" || -z "${CONDA_PREFIX:-}" ]]; then
    printf '%s\n' 'KCF ROS: missing deployment environment configuration; run make kcf-setup.' >&2
    return 1
fi
_kcf_ros_root="$(cd -- "$DEPLOY_ROOT_DIR" && pwd)"
_kcf_ros_workspace="${KCF_ROS_WORKSPACE:-$_kcf_ros_root/.cache/ros_py311}"
if [[ ! -f "$_kcf_ros_workspace/install/local_setup.bash" ]]; then
    printf '%s\n' 'KCF ROS bindings are missing; run make kcf-setup in deploy_robot.' >&2
    unset _kcf_ros_root _kcf_ros_workspace
    return 1
fi

declare -gA _KCF_ROS_CONDA_SAVED=()
declare -ga _KCF_ROS_CONDA_VARS=(
    AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH
    PKG_CONFIG_PATH PYTHONPATH PYTHONNOUSERSITE ROS_DISTRO ROS_PYTHON_VERSION
    ROS_VERSION ROS_ETC_DIR ROS_DOMAIN_ID ROS_LOCALHOST_ONLY ROS_LOG_DIR
    DEPLOY_ROOT_DIR
)
for _kcf_ros_name in "${_KCF_ROS_CONDA_VARS[@]}"; do
    if [[ -v "$_kcf_ros_name" ]]; then
        _KCF_ROS_CONDA_SAVED["$_kcf_ros_name"]="${!_kcf_ros_name}"
    fi
done
_KCF_ROS_CONDA_ACTIVE=1
_kcf_ros_conda_path="$PATH"
source /opt/ros/humble/setup.bash
source "$_kcf_ros_workspace/install/local_setup.bash"
# ROS setup prepends its Python 3.10 ros2 launcher. The matching ros2 wrapper
# is installed in Conda's bin instead, so leave PATH entirely to Conda.
export PATH="$_kcf_ros_conda_path"
export PYTHONNOUSERSITE=1
export ROS_LOG_DIR="$_kcf_ros_workspace/log/ros"
export ROS_DOMAIN_ID="${KCF_ROS_DOMAIN_ID:-73}"
export ROS_LOCALHOST_ONLY="${KCF_ROS_LOCALHOST_ONLY:-1}"
if ! "$CONDA_PREFIX/bin/python" -c 'import sys,numpy,rclpy; from pathlib import Path; from rclpy.impl.implementation_singleton import rclpy_implementation; assert sys.version_info[:2] == (3,11) and int(numpy.__version__.split(".")[0]) == 2; assert Path(rclpy_implementation.__file__).is_relative_to(Path(sys.argv[1])/"install"), "Wrong ROS bindings"' "$_kcf_ros_workspace"; then
    printf '%s\n' 'KCF ROS activation failed; run make kcf-setup to repair the environment.' >&2
    source "$DEPLOY_ROOT_DIR/tools/ros_py311/conda_deactivate.sh"
    unset _kcf_ros_root _kcf_ros_workspace _kcf_ros_conda_path _kcf_ros_name
    return 1
fi
unset _kcf_ros_root _kcf_ros_workspace _kcf_ros_conda_path _kcf_ros_name
