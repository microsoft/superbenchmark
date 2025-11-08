FROM nvcr.io/nvidia/pytorch:25.02-py3

# OS:
#   - Ubuntu: 24.04
#   - OpenMPI: 4.1.7+
#   - Docker Client: 20.10.8
# NVIDIA:
#   - CUDA: 12.8.0.38
#   - cuDNN: 9.7.1.26
#   - cuBLAS: 12.8.3.14
#   - NCCL: v2.25.1
#   - TransformerEngine 2.0
# Mellanox:
#   - MOFED_VERSION; 5.4-rdmacore39.0
#   - HPC-X: v2.21.0-CUDA12.x
# Intel:
#   - mlc: v3.12

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
    libncurses-dev \
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
    rsync \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Install CMake 3.30.4 for nvbench compatibility
RUN apt-get update && \
    apt-get remove -y cmake cmake-data && \
    apt-get autoremove -y && \
    cd /tmp && \
    ARCH=$(uname -m) && \
    case ${ARCH} in \
        "aarch64") CMAKE_ARCH="aarch64" ;; \
        "x86_64") CMAKE_ARCH="x86_64" ;; \
        "arm64") CMAKE_ARCH="aarch64" ;; \
        *) CMAKE_ARCH="x86_64" ;; \
    esac && \
    echo "Detected architecture: ${ARCH}, using CMAKE_ARCH: ${CMAKE_ARCH}" && \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-${CMAKE_ARCH}.tar.gz && \
    tar -xzf cmake-3.30.4-linux-${CMAKE_ARCH}.tar.gz && \
    mv cmake-3.30.4-linux-${CMAKE_ARCH} /opt/cmake && \
    ln -sf /opt/cmake/bin/* /usr/local/bin/ && \
    rm -rf cmake-3.30.4-linux-${CMAKE_ARCH}* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG NUM_MAKE_JOBS=
ARG TARGETPLATFORM
ARG TARGETARCH

# Install Docker
ENV DOCKER_VERSION=20.10.8
RUN TARGETARCH_HW=$(uname -m) && \
    wget -q https://download.docker.com/linux/static/stable/${TARGETARCH_HW}/docker-${DOCKER_VERSION}.tgz -O docker.tgz && \
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
ENV OFED_VERSION=24.10-1.1.4.0
RUN TARGETARCH_HW=$(uname -m) && \
    cd /tmp && \
    wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu24.04-${TARGETARCH_HW}.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu24.04-${TARGETARCH_HW}.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu24.04-${TARGETARCH_HW}/mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install HPC-X
ENV HPCX_VERSION=v2.21
RUN TARGETARCH_HW=$(uname -m) && \
    cd /opt && \
    rm -rf hpcx && \
    wget https://content.mellanox.com/hpc/hpc-x/${HPCX_VERSION}/hpcx-${HPCX_VERSION}-gcc-doca_ofed-ubuntu24.04-cuda12-${TARGETARCH_HW}.tbz -O hpcx.tbz && \
    tar xf hpcx.tbz && \
    mv hpcx-${HPCX_VERSION}-gcc-doca_ofed-ubuntu24.04-cuda12-${TARGETARCH_HW} hpcx && \
    rm hpcx.tbz

# Installs specific to amd64 platform
RUN if [ "$TARGETARCH" = "amd64" ]; then \
    # Install Intel MLC
    cd /tmp && \
    wget -q https://downloadmirror.intel.com/866182/mlc_v3.12.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz && \
    # Install AOCC compiler
    wget https://download.amd.com/developer/eula/aocc-compiler/aocc-compiler-4.0.0_1_amd64.deb && \
    apt install -y ./aocc-compiler-4.0.0_1_amd64.deb && \
    rm -rf aocc-compiler-4.0.0_1_amd64.deb && \
    # Install AMD BLIS
    wget https://download.amd.com/developer/eula/blis/blis-4-0/aocl-blis-linux-aocc-4.0.tar.gz && \
    tar xzf aocl-blis-linux-aocc-4.0.tar.gz && \
    mv amd-blis /opt/AMD && \
    rm -rf aocl-blis-linux-aocc-4.0.tar.gz; \
    else \
    echo "Skipping Intel MLC, AOCC and AMD Bliss installations for non-amd64 architecture: $TARGETARCH"; \
    fi

# Install NCCL 2.25.1
RUN cd /tmp && \
    git clone -b v2.25.1-1 https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j ${NUM_MAKE_JOBS} src.build \
    NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100 \
    -gencode=arch=compute_90,code=sm_90 \
    -gencode=arch=compute_80,code=sm_80" && \
    make install && \
    rm -rf /tmp/nccl

# Install UCX with multi-threading support
ENV UCX_VERSION=1.18.0
RUN cd /tmp && \
    wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}-rc1/ucx-${UCX_VERSION}.tar.gz && \
    tar xzf ucx-${UCX_VERSION}.tar.gz && \
    cd ucx-${UCX_VERSION} && \
    ./contrib/configure-release-mt --prefix=/usr/local && \
    make -j ${NUM_MAKE_JOBS} && \
    make install

ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/mpi/lib:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment && \
    echo "source /opt/hpcx/hpcx-init.sh && hpcx_load" | tee -a /etc/bash.bashrc >> /etc/profile.d/10-hpcx.sh

# Add config files
ADD dockerfile/etc /opt/microsoft/

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN make -C third_party cuda_with_msccl cuda_nvbench

ADD . .
RUN python3 -m pip install --upgrade setuptools==70.3.0 && \
    python3 -m pip install --no-cache-dir .[nvworker] && \
    make cppbuild && \
    make postinstall && \
    rm -rf .git
