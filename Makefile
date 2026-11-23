# deploy_robot environment setup
ENV := deploy
.DEFAULT_GOAL := install

.PHONY: install update uninstall kcf-setup hardware-setup

# Build the isolated bindings and install automatic activation/deactivation.
kcf-setup:
	conda run --no-capture-output -n $(ENV) bash tools/ros_py311/prepare_ros.bash
	conda run --no-capture-output -n $(ENV) bash tools/ros_py311/build_ros.bash all
	conda run --no-capture-output -n $(ENV) bash tools/ros_py311/install_hooks.bash

# Install the official SDK and native DDS library into this Conda environment.
# The SDK wheel omits its CRC shared libraries, so install those from the same pinned source.
hardware-setup:
	conda run --no-capture-output -n $(ENV) bash -c 'set -euo pipefail; \
		deploy_sdk_prefix="$${CONDA_PREFIX:?}"; \
		deploy_sdk_build=$$(mktemp -d); trap '\''rm -rf "$$deploy_sdk_build"'\'' EXIT; \
		if [ ! -f "$$deploy_sdk_prefix/lib/libddsc.so" ]; then \
			git clone --quiet --depth 1 --branch 0.10.5 https://github.com/eclipse-cyclonedds/cyclonedds.git "$$deploy_sdk_build/dds"; \
			cmake -S "$$deploy_sdk_build/dds" -B "$$deploy_sdk_build/build" \
				-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_TYPE=Release \
				-DCMAKE_INSTALL_PREFIX="$$deploy_sdk_prefix" -DBUILD_EXAMPLES=OFF \
				-DBUILD_TESTING=OFF -DENABLE_SECURITY=OFF -DENABLE_SSL=NO; \
			cmake --build "$$deploy_sdk_build/build" --parallel 4 --target install; \
		fi; \
		git clone --quiet https://github.com/unitreerobotics/unitree_sdk2_python.git "$$deploy_sdk_build/sdk"; \
		git -C "$$deploy_sdk_build/sdk" checkout --quiet --detach 814556d15970dd2ecf1c9984e845ca02ab07e206; \
		CYCLONEDDS_HOME="$$deploy_sdk_prefix" python -m pip install --no-cache-dir "$$deploy_sdk_build/sdk"; \
		deploy_sdk_lib=$$(python -c '\''from pathlib import Path; import unitree_sdk2py.utils; print(Path(unitree_sdk2py.utils.__file__).parent / "lib")'\''); \
		mkdir -p "$$deploy_sdk_lib"; \
		cp "$$deploy_sdk_build/sdk/unitree_sdk2py/utils/lib/"*.so "$$deploy_sdk_lib/"; \
		python -c '\''from unitree_sdk2py.utils.crc import CRC; CRC(); print("Unitree SDK and CRC library ready")'\'''

# create the conda env and set DEPLOY_ROOT_DIR to this repo
install:
	conda env create -f environment.yml
	conda env config vars set -n $(ENV) DEPLOY_ROOT_DIR="$(CURDIR)"
	$(MAKE) kcf-setup

# update the conda env from environment.yml and refresh the root path
update:
	conda env update -f environment.yml --prune
	conda env config vars set -n $(ENV) DEPLOY_ROOT_DIR="$(CURDIR)"
	$(MAKE) kcf-setup

# remove the conda env (must not be active; conda refuses to remove the current env)
uninstall:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV)" ]; then \
		echo "Env '$(ENV)' is active. Run 'conda deactivate' first, then 'make uninstall'."; \
	else \
		conda env remove -n $(ENV) -y; \
	fi
