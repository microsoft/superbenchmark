# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM ubuntu:18.04

# Install the python3.7 and pip
RUN apt-get update && apt-get install -y \
    python3.7-dev \
    python3-pip

# Change default python3 version to python3.7
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 10

# Upgrade pip and instll flake8 and pytest 
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install flake8 pytest

# Create workspace
WORKDIR /superbench
COPY . /superbench

# Check code format using flake8 
RUN python3 -m flake8
    
# Install library and run pytest
RUN python3 -m pytest -v
