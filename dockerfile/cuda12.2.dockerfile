FROM nvcr.io/nvidia/pytorch:23.10-py3

# OS:
#   - Ubuntu: 22.04
#   - OpenMPI: 4.1.5rc2
#   - Docker Client: 20.10.8
# NVIDIA:
#   - CUDA: 12.2.2
#   - cuDNN: 8.9.5
#   - NCCL: v2.18.3-1
# Mellanox:
#   - OFED: 23.07-0.5.1.2
#   - HPC-X: v2.16
# Intel:
#   - mlc: v3.11

LABEL maintainer="SuperBench"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    bc \
    build-essential \
    curl \
    dmidecode \
    ffmpeg \
    git \
    iproute2 \
    jq \
    libaio-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libboost-program-options-dev \
    libcap2 \
    libcurl4-openssl-dev \
    libnuma-dev \
    libpci-dev \
    libswresample-dev \
    libtinfo5 \
    libtool \
    lshw \
    python3-mpi4py \
    net-tools \
    nlohmann-json3-dev \
    openssh-client \
    openssh-server \
    pciutils \
    sudo \
    util-linux \
    vim \
    wget \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ARG NUM_MAKE_JOBS=

# Install Docker
ENV DOCKER_VERSION=20.10.8
RUN cd /tmp && \
    wget -q https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz -O docker.tgz && \
    tar --extract --file docker.tgz --strip-components 1 --directory /usr/local/bin/ && \
    rm docker.tgz

# Update system config
RUN mkdir -p /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i "s/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*PermitUserEnvironment no/PermitUserEnvironment yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*Port.*/Port 22/" /etc/ssh/sshd_config && \
    echo "* soft nofile 1048576\n* hard nofile 1048576" >> /etc/security/limits.conf && \
    echo "root soft nofile 1048576\nroot hard nofile 1048576" >> /etc/security/limits.conf

# Install OFED
ENV OFED_VERSION=23.07-0.5.1.2
RUN cd /tmp && \
    wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu22.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu22.04-x86_64.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu22.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install HPC-X
ENV HPCX_VERSION=v2.16
RUN cd /opt && \
    rm -rf hpcx && \
    wget -q https://content.mellanox.com/hpc/hpc-x/${HPCX_VERSION}/hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64.tbz -O hpcx.tbz && \
    tar xf hpcx.tbz && \
    mv hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64 hpcx && \
    rm hpcx.tbz

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/793041/mlc_v3.11.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Install AOCC compiler
RUN cd /tmp && \
    wget https://download.amd.com/developer/eula/aocc-compiler/aocc-compiler-4.0.0_1_amd64.deb && \
    apt install -y ./aocc-compiler-4.0.0_1_amd64.deb && \
    rm -rf aocc-compiler-4.0.0_1_amd64.deb

# Install AMD BLIS
RUN cd /tmp && \
    wget https://download.amd.com/developer/eula/blis/blis-4-0/aocl-blis-linux-aocc-4.0.tar.gz && \
    tar xzf aocl-blis-linux-aocc-4.0.tar.gz && \
    mv amd-blis /opt/AMD && \
    rm -rf aocl-blis-linux-aocc-4.0.tar.gz

# Install NCCL 2.18.3
RUN cd /tmp && \
    git clone -b v2.18.3-1 https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j src.build && \
    make install && \
    rm -rf /tmp/nccl

ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment && \
    echo "source /opt/hpcx/hpcx-init.sh && hpcx_load" >> /etc/bash.bashrc

# Add config files
ADD dockerfile/etc /opt/microsoft/

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN make -C third_party cuda_with_msccl

ADD . .
RUN python3 -m pip install --upgrade setuptools==65.7 && \
    python3 -m pip install --no-cache-dir .[nvworker] && \
    make cppbuild && \
    make postinstall && \
    rm -rf .git
