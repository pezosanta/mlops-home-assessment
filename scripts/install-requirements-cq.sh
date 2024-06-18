#!/bin/bash

# Setting the relative path of current script's parent directory and the project's root directory into a local variable
# This way, we can call this script from any directory and it will still work
CURRENT_SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJECT_DIR="$CURRENT_SCRIPT_DIR"/../

# Upgrading pip
python -m pip install --upgrade pip

# Installing code quality related packages and pre-commit hooks
cd "$PROJECT_DIR" || { echo "Could not cd to $PROJECT_DIR"; exit 1; }
poetry install --only test --no-root
poetry run pre-commit install
