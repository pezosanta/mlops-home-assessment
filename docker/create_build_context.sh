#!/bin/bash

# Exit bash whenever an error occurs
set -e

workSpace=$(dirname "${BASH_SOURCE[0]}")
projectDir="$workSpace"/../

# Exporting the Python package dependencies into requirements files
poetry export --format requirements.txt --output "$workSpace"/requirements-train.txt --with train
poetry export --format requirements.txt --output "$workSpace"/requirements-prod.txt --with prod

# Building the Python package (creating wheel file)
# shellcheck disable=SC2154
poetry version "$version"
poetry build
poetry version 0.0.0

cp "$projectDir"/dist/*.whl "$workSpace"
