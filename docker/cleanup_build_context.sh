#!/bin/bash

# Exit bash whenever an error occurs
set -e

workSpace=`dirname ${BASH_SOURCE[0]}`
projectDir=$workSpace/../

rm $workSpace/requirements-prod.txt $workSpace/requirements-prod.sh $workSpace/*.whl
rm -r $projectDir/dist $projectDir/*.egg-info
