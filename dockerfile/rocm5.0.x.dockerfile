ARG BASE_IMAGE=rocm/pytorch:rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.9.0
FROM ${BASE_IMAGE}

# 5.1.x base images:
# rocm5.0   - rocm/pytorch:rocm5.0_ubuntu18.04_py3.7_pytorch_1.9.0
# rocm5.0.1 - rocm/pytorch:rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.9.0

# OS:
#   - Ubuntu: 18.04
#   - OpenMPI: 4.0.5
#   - Docker Client: 20.10.8
# ROCm:
#   - ROCm: 5.0.x
#   - RCCL: 2.10.3
#   - RCCL RDMA SHARP plugins: rocm-rel-5.0
#   - hipify: 5.0.x
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
    libnuma-dev \
    libpci-dev \
    libtinfo5 \
    libtool \
    lshw \
    net-tools \
    numactl \
    openssh-client \
    openssh-server \
    pciutils \
    rsync \
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
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tgz && \
    PATH=/usr/bin:${PATH} MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
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

# Install rccl-rdma-sharp-plugins
ENV SHARP_VERSION=5.0
RUN cd /opt/rocm && \
    git clone -b release/rocm-rel-${SHARP_VERSION} https://github.com/ROCmSoftwarePlatform/rccl-rdma-sharp-plugins.git && \
    cd rccl-rdma-sharp-plugins && \
    ./autogen.sh && ./configure --prefix=/usr/local && make -j ${NUM_MAKE_JOBS} && make install

ENV PATH="${PATH}:/opt/rocm/hip/bin/" \
    LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN make -C third_party rocm

ADD . .
RUN python3 -m pip install .[torch,ort]  && \
    make cppbuild && \
    make postinstall
