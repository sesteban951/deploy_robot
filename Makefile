# deploy_robot environment setup
ENV := deploy
.DEFAULT_GOAL := install

.PHONY: install update uninstall kcf-setup

# Build the isolated bindings and install automatic activation/deactivation.
kcf-setup:
	conda run --no-capture-output -n $(ENV) bash tools/ros_py311/prepare_ros.bash
	conda run --no-capture-output -n $(ENV) bash tools/ros_py311/build_ros.bash all
	conda run --no-capture-output -n $(ENV) bash tools/ros_py311/install_hooks.bash

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
