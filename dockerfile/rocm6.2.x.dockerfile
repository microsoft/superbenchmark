ARG BASE_IMAGE=rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.3.0

FROM ${BASE_IMAGE}

# OS:
#   - Ubuntu: 22.04
#   - Docker Client: 20.10.8
# ROCm:
#   - ROCm: 6.2
# Lib:
#   - torch: 2.3.0
#   - rccl: 2.18.3+hip6.0 develop:7e1cbb4
#   - hipblaslt: release-staging/rocm-rel-6.2
#   - rocblas: release-staging/rocm-rel-6.2
#   - openmpi: 4.1.x
# Intel:
#   - mlc: v3.11

LABEL maintainer="SuperBench"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -q install -y --no-install-recommends  \
    autoconf \
    automake \
    bc \
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
    libcurl4-openssl-dev \
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
    python3-mpi4py \
    rsync \
    sudo \
    util-linux \
    vim \
    wget \
    && \
    rm -rf /tmp/*

ARG NUM_MAKE_JOBS=64

# Check if CMake is installed and its version
RUN cmake_version=$(cmake --version 2>/dev/null | grep -oP "(?<=cmake version )(\d+\.\d+)" || echo "0.0") && \
    required_version="3.24.1" && \
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
    echo "CMake version is greater than or equal to 3.24.1"; \
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

ENV ROCM_PATH=/opt/rocm

# Install OpenMPI
ENV OPENMPI_VERSION=4.1.x
ENV MPI_HOME=/usr/local/mpi
# Check if Open MPI is installed
RUN cd /tmp && \
    git clone --recursive https://github.com/open-mpi/ompi.git -b v${OPENMPI_VERSION}  && \
    cd ompi && \
    ./autogen.pl && \
    mkdir build && \
    cd build && \
    ../configure --prefix=/usr/local/mpi  --enable-orterun-prefix-by-default --enable-mpirun-prefix-by-default  --enable-prte-prefix-by-default --with-rocm=/opt/rocm && \
    make -j $(nproc) && \
    make -j $(nproc) install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/openmpi-${OPENMPI_VERSION}*

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/763324/mlc_v3.10.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Install RCCL
RUN cd /opt/ &&  \
    git clone -b release/rocm-rel-6.2 https://github.com/ROCmSoftwarePlatform/rccl.git && \
    cd rccl && \
    mkdir build && \
    cd build && \
    CXX=/opt/rocm/bin/hipcc cmake -DHIP_COMPILER=clang -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}/hsa;${ROCM_PATH}/hip;${ROCM_PATH}/share/rocm/cmake/;${ROCM_PATH}" \
    .. && \
    make -j${NUM_MAKE_JOBS}

# Install AMD SMI Python Library
RUN apt install amd-smi-lib -y && \
    cd /opt/rocm/share/amd_smi && \
    python3 -m pip install .

ENV PATH="/usr/local/mpi/bin:/opt/superbench/bin:/usr/local/bin/:/opt/rocm/hip/bin/:/opt/rocm/bin/:${PATH}" \
    LD_PRELOAD="/opt/rccl/build/librccl.so:$LD_PRELOAD" \
    LD_LIBRARY_PATH="/usr/local/mpi/lib:/usr/lib/x86_64-linux-gnu/:/usr/local/lib/:/opt/rocm/lib:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment

RUN apt install rocm-cmake -y && \
    python3 -m pip install --upgrade pip wheel setuptools==65.7

WORKDIR ${SB_HOME}

ADD third_party third_party
# Apply patch
RUN cd third_party/perftest && \
    git apply ../perftest_rocm6.patch
RUN make RCCL_HOME=/opt/rccl/build/ ROCBLAS_BRANCH=release-staging/rocm-rel-6.2 HIPBLASLT_BRANCH=release-staging/rocm-rel-6.2 ROCM_VER=rocm-5.5.0 -C third_party rocm -o cpu_hpl -o cpu_stream -o megatron_lm
RUN cp -r /opt/superbench/third_party/hipBLASLt/build/release/hipblaslt-install/lib/*  /opt/rocm/lib/ && \
    cp -r /opt/superbench/third_party/hipBLASLt/build/release/hipblaslt-install/include/*  /opt/rocm/include/
RUN cd third_party/Megatron/Megatron-DeepSpeed && \
    git apply ../megatron_deepspeed_rocm6.patch

# Install transformer_engine
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && \
    export NVTE_FRAMEWORK=pytorch && \
    pip install .

ADD . .
ENV USE_HIP_DATATYPE=1
ENV USE_HIPBLAS_COMPUTETYPE=1
RUN python3 -m pip install .[amdworker]  && \
    CXX=/opt/rocm/bin/hipcc make cppbuild  && \
    make postinstall

