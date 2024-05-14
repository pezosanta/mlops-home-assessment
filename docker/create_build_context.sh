#!/bin/bash

# Exit bash whenever an error occurs
set -e

workSpace=`dirname ${BASH_SOURCE[0]}`
projectDir=$workSpace/../
requirementsDir=$projectDir/../

# Building the Python package (creating wheel file)
python -m build

cp $requirementsDir/requirements-prod.txt $requirementsDir/requirements-prod.sh $workSpace
cp $projectDir/dist/*.whl $workSpace
