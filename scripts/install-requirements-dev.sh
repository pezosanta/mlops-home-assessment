#!/bin/bash

# Setting the relative path of current script's parent directory into a local variable
# This way, we can call this script from any directory and it will still work
CURRENT_SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

# Upgrading pip
python -m pip install --upgrade pip

# Installing Python package build related packages
source "$CURRENT_SCRIPT_DIR"/install-requirements-build.sh

# Installing code quality related packages and pre-commit hooks
source "$CURRENT_SCRIPT_DIR"/install-requirements-cq.sh

# Installing root (in editable mode) + main, train, dev and prod package dependencies
poetry install --without test
