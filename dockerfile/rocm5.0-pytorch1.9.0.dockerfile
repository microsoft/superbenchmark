FROM rocm/pytorch:rocm5.0_ubuntu18.04_py3.7_pytorch_1.9.0

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
    libpci-dev \
    libtinfo5 \
    libtool \
    lshw \
    net-tools \
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
RUN cd /tmp && wget https://github.com/openucx/ucx/releases/download/v1.12.0/ucx-1.12.0.tar.gz && \
    tar xzf ucx-1.12.0.tar.gz && \
    cd ucx-1.12.0 && \
    mkdir build && cd build/ && \
    ../contrib/configure-release --prefix=/opt/ucx && \
    make -j $(nproc) && make install && \
    rm -rf /tmp/ucx-1.12.0

# Install OpenMPI
ENV OPENMPI_VERSION=4.0.5
RUN cd /tmp && \
    wget -q https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
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
    wget --user-agent="Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0" https://www.intel.com/content/dam/develop/external/us/en/documents/mlc_v3.9a.tgz && \
    tar xvf mlc_v3.9a.tgz && \
    cp ./Linux/mlc /usr/local/bin/ && \
    cd /tmp && \
    rm -rf mlc

# Install rccl-rdma-sharp-plugins
RUN cd /opt/rocm && \
    git clone -b release/rocm-rel-5.0 https://github.com/ROCmSoftwarePlatform/rccl-rdma-sharp-plugins.git && \
    cd rccl-rdma-sharp-plugins && \
    ./autogen.sh && ./configure && make -j64

# Install llvm+clang, required by hipify
RUN cd /tmp && git clone -b llvmorg-13.0.0 https://github.com/llvm/llvm-project.git && \
    cd llvm-project/&& \
    mkdir build && cd build &&  \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/lib/llvm-13 -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_PROJECTS="clang"  -DCMAKE_BUILD_TYPE=Release ../llvm && \
    make -j $(nproc) install

# Install hipify
RUN cd /tmp && git clone -b rocm-5.0.0 https://github.com/ROCm-Developer-Tools/HIPIFY.git && \
    cd HIPIFY && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=Release  -DCMAKE_INSTALL_PREFIX=/opt/rocm/hipify -DCMAKE_PREFIX_PATH=/usr/lib/llvm-13 ..  && \
    make -j $(nproc) install

ENV PATH="${PATH}:/opt/rocm/hip/bin/:/opt/rocm/hipify/" \
    LD_LIBRARY_PATH="/opt/rocm/rccl-rdma-sharp-plugins/src/.libs:/usr/local/lib/:${LD_LIBRARY_PATH}" \
    SB_HOME="/opt/superbench" \
    SB_MICRO_PATH="/opt/superbench"

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN echo gfx90a >> /opt/rocm/bin/target.lst && ROCM_VERSION=rocm-5.0.0 make -j -C third_party rocm

ADD . .
RUN python3 -m pip install .[torch] && \
    make cppbuild
