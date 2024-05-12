#!/bin/bash

# Setting the relative path of current script's parent directory into an environment variable
CURRENT_SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`

# Installing code quality related packages and pre-commit hooks
source $CURRENT_SCRIPT_DIR/requirements-cq.sh

# Installing development package dependencies
pip install -r $CURRENT_SCRIPT_DIR/requirements-dev.txt

# Installing the image-classifier package in editable mode
pip install -v -e $CURRENT_SCRIPT_DIR
