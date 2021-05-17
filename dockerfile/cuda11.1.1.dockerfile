FROM nvcr.io/nvidia/pytorch:20.12-py3

LABEL maintainer="SuperBench"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    jq \
    vim \
    git \
    curl \
    wget \
    lshw \
    dmidecode \
    util-linux \
    automake \
    autoconf \
    libtool \
    net-tools \
    openssh-client \
    openssh-server \
    pciutils \
    libpci-dev \
    libaio-dev \
    libcap2 \
    libtinfo5

# Install CMake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.sh \
    -O /tmp/cmake-install.sh && \
    chmod +x /tmp/cmake-install.sh && \
    mkdir /usr/local/cmake && \
    /tmp/cmake-install.sh --skip-license --prefix=/usr/local/cmake && \
    rm /tmp/cmake-install.sh

# Configure SSH
RUN mkdir -p /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i "s/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*Port.*/Port 22/" /etc/ssh/sshd_config && \
    echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config

# Install OpenMPI
ARG OMPI_VERSION=4.0.5
RUN cd /tmp && \
    wget -q https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-${OMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OMPI_VERSION}.tar.gz && \
    cd openmpi-${OMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi-${OMPI_VERSION}*

# Install OFED
ARG OFED_VERSION=5.2-1.0.4.0
RUN cd /tmp && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install OFED perftest
RUN git clone -b v4.5-0.2 https://github.com/linux-rdma/perftest.git /usr/local/perftest && \
    cd /usr/local/perftest && \
    ./autogen.sh && \
    ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && \
    make -j && \
    make install

# Install NCCL
RUN git clone -b v2.8.4-1 https://github.com/NVIDIA/nccl /usr/local/nccl && \
    cd /usr/local/nccl && \
    make -j src.build && \
    make install
RUN git clone https://github.com/nvidia/nccl-tests /usr/local/nccl-tests && \
    cd /usr/local/nccl-tests && \
    make MPI=1 MPI_HOME=/usr/local/mpi/ -j

ENV PATH="${PATH}:/usr/local/cmake/bin:/usr/local/nccl-tests/build" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    SB_HOME="/opt/superbench"

WORKDIR ${SB_HOME}
ADD . .

RUN cd ${SB_HOME} && \
    python3 -m pip install .[torch]
