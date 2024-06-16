#!/bin/bash

# Setting the relative path of current script's parent directory and the project's root directory into a local variable
CURRENT_SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
PROJECT_DIR="$CURRENT_SCRIPT_DIR"/../

# Installing Poetry
cd "$PROJECT_DIR" || { echo "Could not cd to $PROJECT_DIR"; exit 1; }
pip install poetry poetry-plugin-export
