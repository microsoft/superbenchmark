ARG BASE_IMAGE=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1

FROM ${BASE_IMAGE}

# OS:
#   - Ubuntu: 24.04
#   - Docker Client: 29.6.2 (client only)
# ROCm:
#   - ROCm: 7.2
# Lib:
#   - torch: 2.9.1
#   - rccl: release/rocm-rel-7.2
#   - hipblaslt: release/rocm-rel-7.2 (hipblaslt-bench only, against system hipBLASLt)
#   - rocblas: release/rocm-rel-7.2
#   - transformer_engine: v2.10_rocm
#   - openmpi: 4.1.x
# Intel:
#   - mlc: v3.12
# Network:
#   - OFED: 25.10-3.1.8 user-space (via NVIDIA DOCA-Host 3.2.3, matches host)

LABEL maintainer="SuperBench"

# Target GPU architectures for ROCm builds (space-separated). Without an explicit
# target, hipcc defaults to gfx906 when no GPU is present at build time, producing
# kernels that run incorrectly on MI300X/MI355X. The default targets MI300X (gfx942)
# and MI355X (gfx950); other architectures can be selected with, e.g.,
# --build-arg AMDGPU_TARGETS="gfx90a gfx942".
ARG AMDGPU_TARGETS="gfx942 gfx950"
ENV AMDGPU_TARGETS="${AMDGPU_TARGETS}"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -q install -y --no-install-recommends  \
    autoconf \
    automake \
    bc \
    build-essential \
    curl \
    dmidecode \
    flex \
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
    apt-get clean && \
    rm -rf /tmp/*

ARG NUM_MAKE_JOBS=64

# Check if CMake is installed and its version
RUN cmake_version=$(cmake --version 2>/dev/null | awk 'NR == 1 { print $3 }') && \
    cmake_version=${cmake_version:-0.0.0} && \
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
    rm -rf /tmp/cmake-${required_version}* ; \
    else \
    echo "CMake version ${cmake_version} is greater than or equal to ${required_version}"; \
    fi

# Install the Docker CLI client only. SuperBench uses only the `docker` client (docker pull/run/rmi)
# against an external/host daemon; the bundled dockerd/containerd/runc/shims are never used in-container,
# so we extract just docker/docker to shrink the image and cut the CVE surface. Pinned to the latest static release.
ENV DOCKER_VERSION=29.6.2
RUN cd /tmp && \
    wget -q https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz -O docker.tgz && \
    tar --extract --file docker.tgz --strip-components 1 --directory /usr/local/bin/ docker/docker && \
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

# Install OFED (user-space only) from the NVIDIA DOCA-Host repo to match the host stack.
ENV DOCA_VERSION=3.2.3
# Check if ofed_info is present and has a version
RUN if ! command -v ofed_info >/dev/null 2>&1; then \
    echo "OFED not found. Installing DOCA-OFED user-space ${DOCA_VERSION}..."; \
    DOCA_REPO="https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/ubuntu${UBUNTU_VERSION}/x86_64" && \
    curl -fsSL "${DOCA_REPO}/doca_keyring.gpg" -o /usr/share/keyrings/doca_keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/doca_keyring.gpg] ${DOCA_REPO}/ ./" > /etc/apt/sources.list.d/doca.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends doca-ofed-userspace=${DOCA_VERSION}-019000 && \
    apt-get clean ; \
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
    make -j ${NUM_MAKE_JOBS} && \
    make -j ${NUM_MAKE_JOBS} install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/ompi

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/866182/mlc_v3.12.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Set CMAKE_POLICY_VERSION_MINIMUM globally so subprojects (RCCL's mscclpp, etc.) still configure
# with CMake 4.0+, which dropped compatibility for cmake_minimum_required < 3.5.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# Install RCCL
RUN cd /opt/ && \
    git clone -b release/rocm-rel-7.2 https://github.com/ROCmSoftwarePlatform/rccl.git && \
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

# Do NOT LD_PRELOAD librccl.so - it causes segfaults on process exit due to HIP static
# object teardown order. The source-built RCCL is picked up through LD_LIBRARY_PATH instead.
# Do NOT put /usr/lib/x86_64-linux-gnu/ before /opt/rocm/lib - the OFED user-space stack can
# ship an older libhsa-runtime64.so there that conflicts with ROCm's version.
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
# RCCL is built from source above and referenced via RCCL_HOME.
# perftest_rocm6.patch changes are already upstream in the submodule version.
# rocm_hipblaslt: skipped - hipblaslt-bench is built separately below.
# rocm_megatron_lm: skipped (broken upstream - pretrain_deepseek.py missing in rocm_dev branch).
# apex_rocm: skipped - all apex imports in Megatron-DeepSpeed are guarded with try/except,
#   superbench has zero direct apex usage, and PyTorch 2.9 has native fused optimizers/AMP.
RUN make RCCL_HOME=/opt/rccl/build/ ROCBLAS_BRANCH=release/rocm-rel-7.2 HIPBLASLT_BRANCH=release/rocm-rel-7.2 ROCM_VER=rocm-5.5.0 -C third_party rocm -o cpu_hpl -o cpu_stream -o megatron_lm -o rocm_hipblaslt -o rocm_megatron_lm -o apex_rocm

# Build hipblaslt-bench only (not the hipBLASLt library/Tensile kernels) and run it against the
# hipBLASLt shipped in the base image. This avoids the multi-hour, memory-hungry Tensile library
# generation that the ROCm 6.4 image needs. The upstream 7.2 build system pulls in AMD-internal
# "origami" headers and a new tensilelite-host C++ library even for client-only builds, so a
# minimal top-level CMakeLists.txt (dockerfile/etc/hipblaslt-bench-standalone.cmake) replaces it
# and compiles only the bench sources against the system hipBLASLt + LAPACK.
# The deps superbuild builds but does not install LAPACK; install it explicitly so the standalone
# build can find /usr/local/lib/{liblapack.a,libcblas.a,libblas.a}. The deps project's reserved
# "install" target is renamed so it also configures on CMake versions without CMP0037 OLD.
COPY dockerfile/etc/hipblaslt-bench-standalone.cmake /tmp/hipblaslt-bench-standalone.cmake
RUN cd third_party && \
    git clone --depth 1 -b release/rocm-rel-7.2 https://github.com/ROCmSoftwarePlatform/hipBLASLt.git && \
    cp /tmp/hipblaslt-bench-standalone.cmake hipBLASLt/CMakeLists.txt && \
    cd hipBLASLt && \
    sed -i '/cmake_policy( SET CMP0037 OLD )/d; s/add_custom_target( install/add_custom_target( hipblaslt_deps_install/' deps/CMakeLists.txt && \
    mkdir -p deps/build && cd deps/build && \
    cmake .. && \
    cmake --build . -j${NUM_MAKE_JOBS} --target lapack && \
    cmake --build lapack/src/lapack-build -j${NUM_MAKE_JOBS} --target install && \
    cd ../.. && \
    hipblaslt_architectures=$(printf '%s' "${AMDGPU_TARGETS}" | tr -s '[:space:]' ';' | sed 's/^;//; s/;$//') && \
    set -- cmake \
        -DCMAKE_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++" \
        -DCMAKE_HIP_COMPILER="${ROCM_PATH}/llvm/bin/clang++" \
        -DCMAKE_HIP_ARCHITECTURES="${hipblaslt_architectures}" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH};/usr/local" \
        -DBLAS_LIBRARIES=/usr/local/lib/libblas.a \
        -DLAPACK_LIBRARIES=/usr/local/lib/liblapack.a \
        -DCMAKE_BUILD_TYPE=Release \
        .. && \
    mkdir -p build && cd build && \
    "$@" && \
    make -j${NUM_MAKE_JOBS} hipblaslt-bench && \
    cp -v hipblaslt-bench /opt/superbench/bin/
RUN cd third_party/Megatron/Megatron-DeepSpeed && \
    git apply ../megatron_deepspeed_rocm6.patch

# Install TransformerEngine - pin to v2.10_rocm, a release line validated on ROCm 7.2 (it bundles
# AOTriton 0.11.2b GPU images for gfx942 and gfx950). CK fused attention stays disabled, as in the
# ROCm 6.4 image. AOTriton is enabled only when every requested target is gfx942 or gfx950,
# and disabled when any other architecture is requested.
# onnxscript/onnx are imported unconditionally by transformer_engine.pytorch; install them with
# pip up front rather than relying on `setup.py install` to resolve them.
RUN transformer_engine_architectures=$(printf '%s' "${AMDGPU_TARGETS}" | tr -s '[:space:]' ';' | sed 's/^;//; s/;$//') && \
    nvte_fused_attn_aotriton=0 && \
    if [ -n "${transformer_engine_architectures}" ]; then \
        nvte_fused_attn_aotriton=1; \
        for arch in $(printf '%s' "${transformer_engine_architectures}" | tr ';' ' '); do \
            case "${arch}" in \
                gfx942|gfx950) ;; \
                *) nvte_fused_attn_aotriton=0 ;; \
            esac; \
        done; \
    fi && \
    python3 -m pip install onnxscript onnx && \
    git clone --recursive -b v2.10_rocm https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && \
    MAX_JOBS="${NUM_MAKE_JOBS}" \
    NVTE_FRAMEWORK=pytorch \
    NVTE_FUSED_ATTN_CK=0 \
    NVTE_FUSED_ATTN_AOTRITON="${nvte_fused_attn_aotriton}" \
    NVTE_ROCM_ARCH="${transformer_engine_architectures}" \
    python3 setup.py install
RUN python3 -c "import transformer_engine.pytorch; print('TE installed successfully')"

ADD . .
ENV USE_HIP_DATATYPE=1
ENV USE_HIPBLAS_COMPUTETYPE=1
RUN python3 -m pip install .[amdworker]  && \
    CXX=/opt/rocm/bin/hipcc make cppbuild  && \
    make postinstall && \
    rm -rf .git

# Fix stale hypothesis plugin from base image (imports removed pkg_resources)
# and add test dependencies missing from the base image.
RUN python3 -m pip install --upgrade hypothesis setuptools pytest-timeout vcrpy
