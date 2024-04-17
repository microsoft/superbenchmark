FROM nvcr.io/nvidia/pytorch:20.12-py3

# OS:
#   - Ubuntu: 20.04
#   - OpenMPI: 4.0.5
#   - Docker Client: 20.10.8
# NVIDIA:
#   - CUDA: 11.1.1
#   - cuDNN: 8.0.5
#   - NCCL: v2.10.3-1
# Mellanox:
#   - OFED: 5.2-2.2.3.0
#   - HPC-X: v2.8.3
#   - NCCL RDMA SHARP plugins: 7cccbc1
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
    libcap2 \
    libnuma-dev \
    libpci-dev \
    libswresample-dev \
    libtinfo5 \
    libtool \
    lshw \
    python3-mpi4py \
    net-tools \
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
    rm -rf /var/lib/apt/lists/* /tmp/* /opt/cmake-3.14.6-Linux-x86_64

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
    make -j ${NUM_MAKE_JOBS} && \
    make install && \
    cd /tmp && \
    rm -rf nccl-rdma-sharp-plugins

# Install NCCL patch
RUN cd /tmp && \
    git clone -b v2.10.3-1 https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j ${NUM_MAKE_JOBS} src.build && \
    make install && \
    cd /tmp && \
    rm -rf nccl

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/793041/mlc_v3.11.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment

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

# Add config files
ADD dockerfile/etc /opt/microsoft/

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN make -C third_party cuda

ADD . .
RUN python3 -m pip install --upgrade setuptools==65.7 && \
    python3 -m pip install --no-cache-dir .[nvworker] && \
    make cppbuild && \
    make postinstall && \
    rm -rf .git
