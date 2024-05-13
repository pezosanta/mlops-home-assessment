#!/bin/bash

# Exit bash whenever an error occurs
set -e

workSpace=`dirname ${BASH_SOURCE[0]}`
projectDir=$workSpace/../

# Building the Python package (creating wheel file)
python -m build

cp $projectDir/requirements-prod.txt $projectDir/requirements-prod.sh $workSpace
cp $projectDir/dist/*.whl $workSpace
