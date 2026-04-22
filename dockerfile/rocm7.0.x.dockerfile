ARG BASE_IMAGE=rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

FROM ${BASE_IMAGE}

# OS:
#   - Ubuntu: 24.04
#   - Docker Client: 20.10.8
# ROCm:
#   - ROCm: 7.0
# Lib:
#   - torch: 2.9.1
#   - rccl: release/rocm-rel-7.0
#   - hipblaslt: release-staging/rocm-rel-7.0
#   - rocblas: release-staging/rocm-rel-7.0
#   - openmpi: 4.1.x
# Intel:
#   - mlc: v3.12

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
    libtinfo6 \
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

# Install CMake via apt if not already present (Ubuntu 24.04 provides >= 3.28)
RUN if ! command -v cmake >/dev/null 2>&1; then \
    apt-get update && apt-get install -y --no-install-recommends cmake; \
    fi && \
    echo "CMake version: $(cmake --version | head -1)"

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


# Set Ubuntu version
ENV UBUNTU_VERSION=24.04

# Install OFED
ENV OFED_VERSION=24.10-1.1.4.0
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
    wget -q https://downloadmirror.intel.com/866182/mlc_v3.12.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Install RCCL
# Set CMAKE_POLICY_VERSION_MINIMUM globally so all subprojects (mscclpp, etc.)
# work with CMake 4.0+ which dropped compat for cmake_minimum_required < 3.5
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN cd /opt/ &&  \
    git clone -b release/rocm-rel-7.0 https://github.com/ROCmSoftwarePlatform/rccl.git && \
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

# Note: Do NOT LD_PRELOAD librccl.so — it causes segfaults on process exit
# due to HIP static object teardown order. Use LD_LIBRARY_PATH instead.
ENV PATH="/usr/local/mpi/bin:/opt/superbench/bin:/usr/local/bin/:/opt/rocm/hip/bin/:/opt/rocm/bin/:${PATH}" \
    LD_LIBRARY_PATH="/opt/rccl/build:/usr/local/mpi/lib:/opt/rocm/lib:/usr/local/lib/:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment

RUN apt install rocm-cmake -y && \
    python3 -m pip install --upgrade pip wheel "setuptools>=69.0"

WORKDIR ${SB_HOME}

ADD third_party third_party
# perftest_rocm6.patch changes are already upstream in the submodule version
# rocm_megatron_lm: broken upstream (pretrain_deepseek.py missing in rocm_dev branch)
# apex_rocm: skipped — all imports guarded, PyTorch 2.9 has native fused optimizers/AMP.
RUN make RCCL_HOME=/opt/rccl/build/ ROCBLAS_BRANCH=release-staging/rocm-rel-7.0 HIPBLASLT_BRANCH=release-staging/rocm-rel-7.0 ROCM_VER=rocm-5.5.0 -C third_party rocm -o cpu_hpl -o cpu_stream -o megatron_lm -o rocm_hipblaslt -o rocm_megatron_lm -o apex_rocm
# Build hipblaslt-bench only (not the library) against system-installed hipBLASLt.
# Build hipblaslt-bench only against system hipBLASLt.
# 7.0 uses HIPBLASLT_USE_ROCROLLER (not ENABLE), BUILD_CLIENTS_BENCHMARKS, Tensile_SKIP_BUILD.
RUN cd third_party && \
    git clone --depth 1 -b release-staging/rocm-rel-7.0 https://github.com/ROCmSoftwarePlatform/hipBLASLt.git && \
    cd hipBLASLt && \
    sed -i '/mxdatagenerator\|mxDataGenerator/d' clients/CMakeLists.txt && \
    sed -i 's/if(OS_RELEASE MATCHES "Ubuntu")/if(FALSE AND OS_RELEASE MATCHES "Ubuntu")/' clients/benchmarks/CMakeLists.txt && \
    sed -i '/add_dependencies(TENSILE_LIBRARY_TARGET rocisa)/d' library/src/amd_detail/rocblaslt/src/CMakeLists.txt && \
    sed -i '/cmake_policy( SET CMP0037 OLD )/d; s/add_custom_target( install/add_custom_target( hipblaslt_deps_install/' deps/CMakeLists.txt && \
    # Pre-build the cblas/lapack dependency (normally done by ./install.sh -d).
    # install.sh -dc builds the full library which we want to skip; build deps standalone.
    mkdir -p deps/build && cd deps/build && \
        CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake .. && \
        cmake --build . -j$(nproc) --target googletest lapack && \
        cmake --build gtest/src/googletest-build -j$(nproc) --target install && \
        cmake --build lapack/src/lapack-build -j$(nproc) --target install && \
        cd ../.. && \
    mkdir -p build/release && cd build/release && \
    CMAKE_POLICY_VERSION_MINIMUM= cmake \
        -DHIPBLASLT_USE_ROCROLLER=OFF \
        -DBUILD_CLIENTS_BENCHMARKS=ON \
        -DBUILD_CLIENTS_TESTS=OFF \
        -DBUILD_CLIENTS_SAMPLES=OFF \
        -DTensile_SKIP_BUILD=ON \
        -DCMAKE_PREFIX_PATH="/opt/rocm;/usr/local" \
        -DCMAKE_BUILD_TYPE=Release \
        ../.. && \
    make -j$(nproc) hipblaslt-bench && \
    cp -v clients/staging/hipblaslt-bench /opt/superbench/bin/
RUN cd third_party/Megatron/Megatron-DeepSpeed && \
    git apply ../megatron_deepspeed_rocm6.patch

# Install TransformerEngine — ROCm 7.0 has hip_fp4.h and gfx950 support,
# so we can use the latest dev branch with full CK fused attention.
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && \
    NVTE_FRAMEWORK=pytorch \
    NVTE_ROCM_ARCH="gfx942;gfx950" \
    python3 setup.py install
RUN python3 -c "import transformer_engine.pytorch; print('TE installed successfully')"

ADD . .
ENV USE_HIP_DATATYPE=1
ENV USE_HIPBLAS_COMPUTETYPE=1
RUN python3 -m pip install .[amdworker]  && \
    CXX=/opt/rocm/bin/hipcc make cppbuild  && \
    make postinstall

# Fix stale hypothesis plugin from base image (imports removed pkg_resources)
# and add test dependencies missing from the base image.
RUN python3 -m pip install --upgrade hypothesis setuptools pytest-timeout vcrpy
