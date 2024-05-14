#!/bin/bash

# Setting the relative path of current script's parent directory into an environment variable
CURRENT_SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
PROJECT_DIR=$CURRENT_SCRIPT_DIR/../

# Installing code quality related packages and pre-commit hooks
pip install -r $CURRENT_SCRIPT_DIR/requirements-cq.txt
cd $PROJECT_DIR && pre-commit install
