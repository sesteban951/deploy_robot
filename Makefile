# deploy_robot environment setup
ENV := deploy

.PHONY: install update uninstall

# create the conda env and set DEPLOY_ROOT_DIR to this repo
install:
	conda env create -f environment.yml
	conda env config vars set -n $(ENV) DEPLOY_ROOT_DIR="$(CURDIR)"

# update the conda env from environment.yml and refresh the root path
update:
	conda env update -f environment.yml --prune
	conda env config vars set -n $(ENV) DEPLOY_ROOT_DIR="$(CURDIR)"

# remove the conda env (must not be active; conda refuses to remove the current env)
uninstall:
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV)" ]; then \
		echo "Env '$(ENV)' is active. Run 'conda deactivate' first, then 'make uninstall'."; \
	else \
		conda env remove -n $(ENV) -y; \
	fi
