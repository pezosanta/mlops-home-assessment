#!/bin/bash

export pythonVersion="python3.10"
export dockerImageTag="image-classifier"

version="$(git describe HEAD --tags --match "*.*.*" | sed 's/\(.*\)-\([0-9]\{1,4\}\)-\(.*\)/\1.dev\2+\3/')"
export version
