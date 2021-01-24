# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

# Install build environment
RUN \
    apt-get update && \
    apt-get install -y apt-utils sudo && \
    apt-get upgrade -y

# Install tools
RUN \
    apt-get install -y vim curl

# Install the gcc-6 and python3.7 and pip
RUN \
    apt-get update -y && \
    apt-get install gcc-6 g++-6 -y && \
    apt-get update -y && \
    apt-get install -y python3.7-dev && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

# Change default python3 version to python3.7
RUN \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 10   

# Install dependency for R3Net
RUN \
    apt-get update && \
    apt-get install -y zip unzip 
        
# Create workspace
WORKDIR /superbench
COPY . /superbench

# Check code format using flake8 
RUN \
    python3 -m pip install flake8 && \
    python3 -m flake8
    
# Install library and run pytest
RUN \
    python3 -m pip install pytest && \
    python3 -m pytest -v