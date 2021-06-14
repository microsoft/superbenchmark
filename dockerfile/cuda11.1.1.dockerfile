FROM nvcr.io/nvidia/pytorch:20.12-py3

LABEL maintainer="SuperBench"

ENV DEBIAN_FRONTEND=noninteractive
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
    libtinfo5 \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /opt/cmake-3.14.6-Linux-x86_64

# Configure SSH
RUN mkdir -p /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i "s/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*Port.*/Port 22/" /etc/ssh/sshd_config && \
    echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config

# Install OFED
ENV OFED_VERSION=5.2-2.2.3.0
RUN cd /tmp && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install HPC-X
RUN cd /opt && \
    wget -q https://azhpcstor.blob.core.windows.net/azhpc-images-store/hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tbz && \
    tar xf hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tbz && \
    ln -s hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64 hpcx && \
    rm hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tbz

# Install NCCL RDMA SHARP plugins
RUN cd /tmp && \
    git clone https://github.com/Mellanox/nccl-rdma-sharp-plugins.git && \
    cd nccl-rdma-sharp-plugins && \
    git reset --hard 7cccbc1 && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local --with-cuda=/usr/local/cuda && \
    make -j && \
    make install && \
    cd /tmp && \
    rm -rf nccl-rdma-sharp-plugins

# Install NCCL patch
RUN cd /tmp && \
    git clone -b bootstrap_tag https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j src.build && \
    make install && \
    cd /tmp && \
    rm -rf nccl

# TODO: move to gitmodules
RUN git clone -b v4.5-0.2 https://github.com/linux-rdma/perftest.git /usr/local/perftest && \
    cd /usr/local/perftest && \
    ./autogen.sh && \
    ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && \
    make -j && \
    make install
RUN git clone https://github.com/nvidia/nccl-tests /usr/local/nccl-tests && \
    cd /usr/local/nccl-tests && \
    make MPI=1 MPI_HOME=/usr/local/mpi/ -j

ENV PATH="/usr/local/nccl-tests/build:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    SB_HOME="/opt/superbench" \
    SB_MICRO_PATH="/opt/superbench"

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN make -j -C third_party

ADD . .
RUN python3 -m pip install .[nvidia,torch] && \
    make cppbuild
