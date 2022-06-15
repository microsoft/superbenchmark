ARG BASE_IMAGE=rocm/pytorch:rocm5.1.3_ubuntu20.04_py3.7_pytorch_1.11.0
FROM ${BASE_IMAGE}

# 5.1.x base images:
# rocm5.1.1 - rocm/pytorch:rocm5.1.1_ubuntu20.04_py3.7_pytorch_1.10.0
# rocm5.1.3 - rocm/pytorch:rocm5.1.3_ubuntu20.04_py3.7_pytorch_1.11.0

# OS:
#   - Ubuntu: 20.04
#   - OpenMPI: 4.0.5
#   - Docker Client: 20.10.8
# ROCm:
#   - ROCm: 5.1.x
#   - RCCL: 2.12.10+6707a27
#   - hipify: 5.1.x
# Mellanox:
#   - OFED: 5.2-2.2.3.0
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
    hipify-clang \
    jq \
    libaio-dev \
    libboost-program-options-dev \
    libcap2 \
    libpci-dev \
    libtinfo5 \
    libtool \
    lshw \
    net-tools \
    libnuma-dev \
    libssl-dev \
    numactl \
    openssh-client \
    openssh-server \
    pciutils \
    util-linux \
    vim \
    wget \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ARG NUM_MAKE_JOBS=

# Upgrade CMake from 3.16 to 3.23
ENV CMAKE_VERSION=3.23.1
ENV CMAKE_REPO="https://github.com/Kitware/CMake/releases/download/v3.23.1/"
RUN wget -nv ${CMAKE_REPO}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar -xvf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap --prefix=/usr --no-system-curl --parallel=16  && \
    make -j16 && \
    sudo make install && \
    cd .. && \
    rm -rf cmake-${CMAKE_VERSION}.tar.gz cmake-${CMAKE_VERSION}

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
ENV UBUNTU_VERSION=20.04
RUN cd /tmp && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
    PATH=/usr/bin:${PATH} MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf MLNX_OFED_LINUX-${OFED_VERSION}*

# Install OpenMPI
ENV OPENMPI_VERSION=4.0.5
RUN cd /tmp && \
    wget -q https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default --with-ucx=/opt/ucx --enable-mca-no-build=btl-uct && \
    make -j ${NUM_MAKE_JOBS} all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi-${OPENMPI_VERSION}*

# Install Intel MLC
RUN cd /tmp && \
    curl https://www.intel.com/content/dam/develop/external/us/en/documents/mlc_v3.9a.tgz -o mlc.tgz && \
    tar xzvf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Install rccl with commitid 6707a27
RUN cd /tmp && \
    git clone https://github.com/ROCmSoftwarePlatform/rccl.git && \
    cd rccl && git checkout 6707a27 && \
    mkdir build && cd build && \
    CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make && make install && \
    cd /tmp && \
    rm -rf rccl

ENV PATH="${PATH}:/opt/rocm/hip/bin/" \
    LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN make ROCBLAS_BRANCH=release/rocm-rel-5.1 -C third_party rocm

ADD . .
RUN python3 -m pip install .[torch,ort]  && \
    make cppbuild && \
    make postinstall
