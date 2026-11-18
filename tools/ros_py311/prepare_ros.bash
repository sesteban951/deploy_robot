#!/usr/bin/env bash
# Prepare the standard isolated KCF ROS Python 3.11 environment; never update the base environment.
set -eo pipefail
kcf_ros_script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
kcf_ros_checkout_root="$(cd -- "$kcf_ros_script_root/../.." && pwd)"
kcf_ros_prepare_root="${KCF_ROS_WORKSPACE:-$kcf_ros_checkout_root/.cache/ros_py311}"
kcf_ros_base_python="${KCF_ROS_BASE_PYTHON:-$(command -v python)}"
"$kcf_ros_base_python" -c 'import sys,numpy,jax,torch; assert sys.version_info[:2] == (3,11) and int(numpy.__version__.split(".")[0]) == 2, "Activate the deploy Python 3.11 / NumPy 2 environment first"'
test -f /opt/ros/humble/setup.bash
command -v /usr/bin/colcon >/dev/null
mkdir -p "$kcf_ros_prepare_root/src"
while IFS=$'\t' read -r kcf_ros_repo kcf_ros_url kcf_ros_tag kcf_ros_commit; do
    if [[ ! -d "$kcf_ros_prepare_root/src/$kcf_ros_repo/.git" ]]; then
        git clone --depth 1 --branch "$kcf_ros_tag" "$kcf_ros_url" "$kcf_ros_prepare_root/src/$kcf_ros_repo"
    fi
    kcf_ros_actual_commit="$(git -C "$kcf_ros_prepare_root/src/$kcf_ros_repo" rev-parse HEAD)"
    if [[ "$kcf_ros_actual_commit" != "$kcf_ros_commit" ]]; then
        printf 'Unexpected source version in %s; not overwriting it.\n' "$kcf_ros_repo" >&2
        exit 1
    fi
done < <("$kcf_ros_base_python" -c 'import json,sys; data=json.load(open(sys.argv[1])); [print(name,source["url"],source["tag"],source["commit"],sep="\t") for name,source in data["sources"].items()]' "$kcf_ros_script_root/ros_py311_sources.json")
if [[ ! -x "$kcf_ros_prepare_root/env/bin/python" ]]; then
    "$kcf_ros_base_python" -m venv --system-site-packages "$kcf_ros_prepare_root/env"
fi
"$kcf_ros_prepare_root/env/bin/python" -c 'import sys,numpy; assert sys.version_info[:2] == (3,11) and int(numpy.__version__.split(".")[0]) == 2'
"$kcf_ros_prepare_root/env/bin/python" -m pip install \
    empy==3.3.4 lark-parser==0.12.0 catkin_pkg==1.1.1 PyYAML==6.0.3 \
    pybind11==2.13.6 netifaces==0.11.0 typeguard==4.6.0
# The unchanged native CUDA evaluator resolves headers/libraries relative to
# sysconfig's active site-packages. Reference existing assets; do not copy or
# reinstall CUDA/JAX, or edit the evaluator to make the overlay work.
kcf_ros_shared_site="$("$kcf_ros_base_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for kcf_ros_asset in nvidia jaxlib; do
    if [[ ! -e "$kcf_ros_prepare_root/env/lib/python3.11/site-packages/$kcf_ros_asset" ]]; then
        ln -s "$kcf_ros_shared_site/$kcf_ros_asset" "$kcf_ros_prepare_root/env/lib/python3.11/site-packages/$kcf_ros_asset"
    fi
done

