# KCF ROS deactivation: restore only exports changed by our activation hook.
# Conda owns PATH and the interpreter; neither is changed here.
if [[ -z "${BASH_VERSION:-}" || -z "${_KCF_ROS_CONDA_ACTIVE:-}" ]]; then
    return 0
fi
for _kcf_ros_name in "${_KCF_ROS_CONDA_VARS[@]}"; do
    if [[ -v "_KCF_ROS_CONDA_SAVED[$_kcf_ros_name]" ]]; then
        printf -v "$_kcf_ros_name" '%s' "${_KCF_ROS_CONDA_SAVED[$_kcf_ros_name]}"
        export "$_kcf_ros_name"
    else
        unset "$_kcf_ros_name"
    fi
done
unset _KCF_ROS_CONDA_ACTIVE _KCF_ROS_CONDA_SAVED _KCF_ROS_CONDA_VARS _kcf_ros_name
