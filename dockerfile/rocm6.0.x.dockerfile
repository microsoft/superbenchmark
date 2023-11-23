ARG BASE_IMAGE=rocm/ms-private:rocm_e2e_gpt4_nightly
FROM ${BASE_IMAGE}

# OS:
#   - Ubuntu: 20.04
#   - OpenMPI: 4.0.5
#   - Docker Client: 20.10.8
# ROCm:
#   - ROCm: 6.0
# Mellanox:
#   - OFED: 5.2-2.2.3.0
# Intel:
#   - mlc: v3.10

LABEL maintainer="SuperBench"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -q install -y --no-install-recommends  \
    autoconf \
    automake \
    build-essential \
    curl \
    dmidecode \
    git \
    hipify-clang \
    iproute2 \
    jq \
    libaio-dev \
    libboost-program-options-dev \
    libcap2 \
    libnuma-dev \
    libpci-dev \
    libssl-dev \
    libtinfo5 \
    libtool \
    lshw \
    net-tools \
    numactl \
    openssh-client \
    openssh-server \
    pciutils \
    rsync \
    sudo \
    util-linux \
    vim \
    wget \
    && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ARG NUM_MAKE_JOBS=

# Check if CMake is installed and its version
RUN cmake_version=$(cmake --version 2>/dev/null | grep -oP "(?<=cmake version )(\d+\.\d+)" || echo "0.0") && \
    required_version="3.26.4" && \
    if [ "$(printf "%s\n" "$required_version" "$cmake_version" | sort -V | head -n 1)" != "$required_version" ]; then \
        echo "existing cmake version is ${cmake_version}" && \
        cd /tmp && \
        wget -q https://github.com/Kitware/CMake/releases/download/v${required_version}/cmake-${required_version}.tar.gz && \
        tar xzf cmake-${required_version}.tar.gz && \
        cd cmake-${required_version} && \
        ./bootstrap --prefix=/usr --no-system-curl --parallel=16 && \
        make -j ${NUM_MAKE_JOBS} && \
        make install && \
        rm -rf /tmp/cmake-${required_version}* \
    else \
        echo "CMake version is greater than or equal to 3.23"; \
    fi

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


# Get Ubuntu version and set as an environment variable
RUN export UBUNTU_VERSION=$(lsb_release -r -s)
RUN echo "Ubuntu version: $UBUNTU_VERSION"
ENV UBUNTU_VERSION=${UBUNTU_VERSION}

# Install OFED
ENV OFED_VERSION=5.9-0.5.6.0
# Check if ofed_info is present and has a version
RUN if ! command -v ofed_info >/dev/null 2>&1; then \
        echo "OFED not found. Installing OFED..."; \
        cd /tmp && \
        wget -q http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
        tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
        PATH=/usr/bin:${PATH} MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
        rm -rf MLNX_OFED_LINUX-${OFED_VERSION}* ; \
    fi

# Install UCX
ENV UCX_VERSION=1.14.1
RUN if [ -z "$(ls -A /opt/ucx)" ]; then \
    echo "/opt/ucx is empty. Installing UCX..."; \
    cd /tmp && \
    git clone https://github.com/openucx/ucx.git -b v${UCX_VERSION} && \
    cd ucx && \
    ./autogen.sh && \
    mkdir build && \
    cd build && \
    ../configure -prefix=$UCX_DIR --with-rocm=/opt/rocm && \
    make -j $(nproc) && make -j $(nproc) install && rm -rf /tmp/ucx-${UCX_VERSION} ; \
    else \
      echo "/opt/ucx is not empty. Skipping UCX installation."; \
    fi

# Install OpenMPI
ENV OPENMPI_VERSION=5.0.x
# Check if Open MPI is installed
RUN [ -d /usr/local/bin/mpirun ] || { \
    echo "Open MPI not found. Installing Open MPI..." && \
    cd /tmp && \
    git clone --recursive https://github.com/open-mpi/ompi.git -b ${OPENMPI_VERSION}  && \
    cd ompi && \
    ./autogen.pl && \
    mkdir build && \
    cd build && \
    ../configure --prefix=/usr/local --with-ucx=/opt/ucx --with-rocm=/opt/rocm && \
    make -j $(nproc) && \
    make -j $(nproc) install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/openmpi-${OPENMPI_VERSION}* ;\
}

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/763324/mlc_v3.10.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Install RCCL
RUN cd /opt/ &&  \
    git clone https://github.com/ROCmSoftwarePlatform/rccl.git && \
    cd rccl && \
    mkdir build && \
    cd build && \
    CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ .. && \
    make -j

ENV PATH="/opt/superbench/bin:/usr/local/bin/:/opt/rocm/hip/bin/:/opt/rocm/bin/:${PATH}" \
    LD_PRELOAD="/opt/rccl/build/librccl.so:$LD_PRELOAD" \
    LD_LIBRARY_PATH="/opt/ucx/lib:/usr/local/lib/:/opt/rocm/lib:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment

WORKDIR ${SB_HOME}

ADD third_party third_party

RUN make RCCL_HOME=/opt/rccl/build/ ROCBLAS_BRANCH=release-staging/rocm-rel-6.0 ROCM_VER=rocm-5.5.0 -C third_party rocm -o cpu_hpl -o cpu_stream

ADD . .
RUN apt install rocm-cmake -y && \
    python3 -m pip install --upgrade wheel setuptools==65.7 && \
    python3 -m pip install .[amdworker]  && \
    make cppbuild && \
    make postinstall

