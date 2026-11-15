# deploy_robot environment setup (foxy branch: Jetson Orin, hardware only)
ENV := deploy

# path to the cloned unitree_sdk2_python repo
UNITREE_SDK_DIR ?= $(HOME)/projects/unitree_sdk2_python
# Cyclone DDS 0.10.2 install prefix (C library built from source, tag 0.10.2).
# NOTE: the system copy in /usr/local (from Unitree's C++ SDK) lacks the idl headers
# that the python cyclonedds package needs to build, so it can't be used here.
CYCLONEDDS_HOME ?= $(HOME)/projects/cyclonedds/install
# ROS2 distro sourced on `conda activate`
ROS_SETUP ?= /opt/ros/foxy/setup.bash

.PHONY: install update uninstall sdk hooks

# create the conda env, set env vars, install the unitree SDK, and add the ROS activation hook
install:
	conda env create -f environment.yml
	conda env config vars set -n $(ENV) DEPLOY_ROOT_DIR="$(CURDIR)" CYCLONEDDS_HOME="$(CYCLONEDDS_HOME)"
	$(MAKE) sdk
	$(MAKE) hooks

# update the conda env from environment.yml and refresh the env vars
update:
	conda env update -f environment.yml --prune
	conda env config vars set -n $(ENV) DEPLOY_ROOT_DIR="$(CURDIR)" CYCLONEDDS_HOME="$(CYCLONEDDS_HOME)"
	$(MAKE) sdk

# (re)install unitree_sdk2py; cyclonedds is rebuilt from source against $(CYCLONEDDS_HOME)
sdk:
	@test -d "$(UNITREE_SDK_DIR)" || { echo "unitree_sdk2_python not found at $(UNITREE_SDK_DIR). Clone it or pass UNITREE_SDK_DIR=<path>."; exit 1; }
	CYCLONEDDS_HOME="$(CYCLONEDDS_HOME)" conda run --no-capture-output -n $(ENV) \
		pip install --no-cache-dir --no-binary cyclonedds "cyclonedds==0.10.2"
	CYCLONEDDS_HOME="$(CYCLONEDDS_HOME)" conda run --no-capture-output -n $(ENV) \
		pip install --no-cache-dir -e "$(UNITREE_SDK_DIR)"

# source ROS2 on `conda activate deploy` so rclpy is importable
hooks:
	@prefix=$$(conda run -n $(ENV) python -c "import sys; print(sys.prefix)"); \
	mkdir -p "$$prefix/etc/conda/activate.d"; \
	printf '# added by deploy_robot Makefile\n[ -f "%s" ] && source "%s"\n' "$(ROS_SETUP)" "$(ROS_SETUP)" \
		> "$$prefix/etc/conda/activate.d/ros.sh"; \
	echo "ROS activation hook written to $$prefix/etc/conda/activate.d/ros.sh"

# remove the conda env (must not be active; conda refuses to remove the current env)
uninstall:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV)" ]; then \
		echo "Env '$(ENV)' is active. Run 'conda deactivate' first, then 'make uninstall'."; \
	else \
		conda env remove -n $(ENV) -y; \
	fi
