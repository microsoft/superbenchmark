FROM rocm/pytorch:rocm5.1.1_ubuntu20.04_py3.7_pytorch_1.10.0

# OS:
#   - Ubuntu: 20.04
#   - OpenMPI: 4.0.5
#   - Docker Client: 20.10.8
# ROCm:
#   - ROCm: 5.1.1
#   - RCCL: 2.11.4
#   - RCCL RDMA SHARP plugins: rocm-rel-5.1
#   - hipify: 5.1.1
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
    mkdir -p mlc && \
    cd mlc && \
    curl https://www.intel.com/content/dam/develop/external/us/en/documents/mlc_v3.9a.tgz -o mlc_v3.9a.tgz && \
    tar xvf mlc_v3.9a.tgz && \
    cp ./Linux/mlc /usr/local/bin/ && \
    cd /tmp && \
    rm -rf mlc

# Install rccl-rdma-sharp-plugins
ENV SHARP_VERSION=5.1
RUN cd /opt/rocm && \
    git clone -b release/rocm-rel-${SHARP_VERSION} https://github.com/ROCmSoftwarePlatform/rccl-rdma-sharp-plugins.git && \
    cd rccl-rdma-sharp-plugins && \
    ./autogen.sh && ./configure --prefix=/usr/local && make -j ${NUM_MAKE_JOBS} && make install

ENV PATH="${PATH}:/opt/rocm/hip/bin/" \
    LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH}" \
    SB_HOME="/opt/superbench" \
    SB_MICRO_PATH="/opt/superbench"

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN ROCM_VERSION=release/rocm-rel-5.1 make -j ${NUM_MAKE_JOBS} -C third_party rocm

ADD . .
RUN python3 -m pip install .[torch,ort]  && \
    make cppbuild
