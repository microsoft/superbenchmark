FROM rocm/pytorch:rocm5.0_ubuntu18.04_py3.7_pytorch_1.9.0

# OS:
#   - Ubuntu: 18.04
#   - OpenMPI: 4.0.5
#   - Docker Client: 20.10.8
# ROCm:
#   - ROCm: 5.0.0
#   - RCCL: 2.10.3
#   - RCCL RDMA SHARP plugins: rocm-rel-5.0
#   - hipify: 5.0.0
# Mellanox:
#   - OFED: 5.2-2.2.3.0
#   - HPC-X: v2.8.3
# Intel:
#   - mlc: v3.9a
# Others:
#   - ucx: 1.12.0
#   - llvm+clang: 1.13.0

LABEL maintainer="SuperBench"

ENV DEBIAN_FRONTEND=noninteractive
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key add - && \
    apt-get update && \
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
    libpci-dev \
    libtinfo5 \
    libtool \
    lshw \
    net-tools \
    libnuma-dev \
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

# Install ucx
ENV UCX_VERSION=1.12.0
RUN cd /tmp && wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz && \
    tar xzf ucx-${UCX_VERSION}.tar.gz && \
    cd ucx-${UCX_VERSION} && \
    mkdir build && cd build/ && \
    ../contrib/configure-release --prefix=/opt/ucx && \
    make -j ${NUM_MAKE_JOBS} && make install && \
    rm -rf /tmp/ucx-${UCX_VERSION}

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

# Install HPC-X
RUN cd /opt && \
    wget -q https://azhpcstor.blob.core.windows.net/azhpc-images-store/hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tbz && \
    tar xf hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tbz && \
    ln -s hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64 hpcx && \
    rm hpcx-v2.8.3-gcc-MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tbz

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
ENV SHARP_VERSION=5.0
RUN cd /opt/rocm && \
    git clone -b release/rocm-rel-${SHARP_VERSION} https://github.com/ROCmSoftwarePlatform/rccl-rdma-sharp-plugins.git && \
    cd rccl-rdma-sharp-plugins && \
    ./autogen.sh && ./configure --prefix=/usr/local && make -j ${NUM_MAKE_JOBS}

# Install llvm+clang, required by hipify
ENV LLVM_VERSION=13.0.0
RUN cd /tmp && git clone -b llvmorg-${LLVM_VERSION} https://github.com/llvm/llvm-project.git && \
    cd llvm-project/&& \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/lib/llvm-${LLVM_VERSION} -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_PROJECTS="clang"  -DCMAKE_BUILD_TYPE=Release ../llvm && \
    make -j ${NUM_MAKE_JOBS} install && \
    cd /tmp && rm -rf llvm-project

# Install hipify
ENV ROCM_VER=rocm-5.0.0
RUN cd /tmp && git clone -b ${ROCM_VER} https://github.com/ROCm-Developer-Tools/HIPIFY.git && \
    cd HIPIFY && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX=/opt/rocm/hipify -DCMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_VERSION} ..  && \
    make -j ${NUM_MAKE_JOBS} install && \
    cd /tmp && rm -rf HIPIFY

ENV PATH="${PATH}:/opt/rocm/hip/bin/:/opt/rocm/hipify/" \
    LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH}" \
    SB_HOME="/opt/superbench" \
    SB_MICRO_PATH="/opt/superbench"

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN ROCM_VERSION=${ROCM_VER} make -j ${NUM_MAKE_JOBS} -C third_party rocm

ADD . .
RUN python3 -m pip install .[torch,ort]  && \
    make cppbuild
