#!/bin/bash

# Exit bash whenever an error occurs
set -e

workSpace=$(dirname "${BASH_SOURCE[0]}")
projectDir="$workSpace"/../

rm -f "$workSpace"/requirements-train.txt "$workSpace"/requirements-prod.txt "$workSpace"/*.whl
rm -rf "$projectDir"/dist
