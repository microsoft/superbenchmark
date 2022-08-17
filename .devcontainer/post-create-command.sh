#!/bin/bash

set -e
export PATH=${HOME}/.local/bin:${PATH}

# install python packages
pip install .[develop,cpuworker] --user --no-cache-dir --progress-bar=off --use-feature=in-tree-build
pre-commit install --install-hooks

# install nodejs packages
cd website
npm install --no-progress
cd -

# try superbench cli
sb
