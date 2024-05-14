#!/bin/bash

# Setting the relative path of current script's parent directory into an environment variable
CURRENT_SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`

# Installing production package dependencies
pip install -r $CURRENT_SCRIPT_DIR/requirements-prod.txt --no-cache-dir
