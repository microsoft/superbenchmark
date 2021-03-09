# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

FROM ubuntu:18.04

# Install the python3.7 and pip
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3.7-dev \
    python3-pip

# Change default python3 version to python3.7
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 10

# Create workspace
WORKDIR /superbench
COPY . /superbench

# Upgrade pip and install dependencies
RUN python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install .[test]

# Install framework
RUN pip3 install \
    torch==1.8.0+cpu \
    torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Lint code
#RUN python3 setup.py lint

# Test code
RUN python3 setup.py test
