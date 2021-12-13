FROM nvcr.io/nvidia/pytorch:20.12-py3

# OS:
#   - Ubuntu: 20.04
#   - OpenMPI: 4.0.5
#   - Docker Client: 20.10.8
# NVIDIA:
#   - CUDA: 11.1.1
#   - cuDNN: 8.0.5
#   - NCCL: bootstrap_tag
# Mellanox:
#   - OFED: 5.2-2.2.3.0
#   - HPC-X: v2.8.3
#   - NCCL RDMA SHARP plugins: 7cccbc1
# Intel:
#   - mlc: v3.9a

LABEL maintainer="SuperBench"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    curl \
    dmidecode \
    git \
    jq \
    libaio-dev \
    libcap2 \
    libnuma-dev \
    libpci-dev \
    libtinfo5 \
    libtool \
    lshw \
    net-tools \
    openssh-client \
    openssh-server \
    pciutils \
    util-linux \
    vim \
    wget \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /opt/cmake-3.14.6-Linux-x86_64

# Install Docker
ENV DOCKER_VERSION=20.10.8
RUN cd /tmp && \
    wget https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz -O docker.tgz && \
    tar --extract --file docker.tgz --strip-components 1 --directory /usr/local/bin/ && \
    rm docker.tgz

# Update system config
RUN mkdir -p /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i "s/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*PermitUserEnvironment no/PermitUserEnvironment yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*Port.*/Port 22/" /etc/ssh/sshd_config && \
    echo -e "* soft nofile 1048576\n* hard nofile 1048576" >> /etc/security/limits.conf && \
    echo -e "root soft nofile 1048576\nroot hard nofile 1048576" >> /etc/security/limits.conf

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

# Install Intel MLC
RUN cd /tmp && \
    mkdir -p mlc && \
    cd mlc && \
    wget --user-agent="Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0" https://www.intel.com/content/dam/develop/external/us/en/documents/mlc_v3.9a.tgz && \
    tar xvf mlc_v3.9a.tgz && \
    cp ./Linux/mlc /usr/local/bin/ && \
    cd /tmp && \
    rm -rf mlc

ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    SB_HOME="/opt/superbench" \
    SB_MICRO_PATH="/opt/superbench"

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN make -j -C third_party cuda

ADD . .
RUN python3 -m pip install .[nvidia,torch,ort] && \
    make cppbuild
