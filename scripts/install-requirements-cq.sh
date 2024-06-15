#!/bin/bash

# Setting the relative path of current script's parent directory into an environment variable
CURRENT_SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJECT_DIR="$CURRENT_SCRIPT_DIR"/../

# Installing Python package build related packages
source "$CURRENT_SCRIPT_DIR"/install-requirements-build.sh

# Installing code quality related packages and pre-commit hooks
cd "$PROJECT_DIR" || { echo "Could not cd to $PROJECT_DIR"; exit 1; }
export POETRY_VIRTUALENVS_CREATE=false
poetry install --only test --no-root
echo "pip list:"
pip list
pre-commit install
