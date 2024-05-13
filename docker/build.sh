#!/bin/bash

# Exit bash whenever an error occurs
set -e

workSpace=`dirname ${BASH_SOURCE[0]}`

# Configure variables
source $workSpace/configure.sh

# Creating the Docker build context
source $workSpace/create_build_context.sh

# Building the Docker image
docker build \
    --build-arg PYTHON_V=$pythonVersion \
    -t $dockerImageTag \
    $workSpace

# Cleanup the Docker build context
source $workSpace/cleanup_build_context.sh
