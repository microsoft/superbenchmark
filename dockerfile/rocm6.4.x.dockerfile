ARG BASE_IMAGE=rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1

FROM ${BASE_IMAGE}

# OS:
#   - Ubuntu: 24.04
#   - Docker Client: 29.6.2 (client only)
# ROCm:
#   - ROCm: 6.4
# Lib:
#   - torch: 2.7.1
#   - rccl: release/rocm-rel-6.4
#   - hipblaslt: release-staging/rocm-rel-6.4
#   - rocblas: release-staging/rocm-rel-6.4
#   - openmpi: 4.1.x
# Intel:
#   - mlc: v3.12
# Network:
#   - OFED: 25.10-3.1.8 user-space (via NVIDIA DOCA-Host 3.2.3, matches host)

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

# Target GPU architecture(s) for ROCm micro-benchmark builds (space-separated).
# Without this, AMDGPU_TARGETS is empty and hipcc defaults to gfx906 at build time
# (no GPU is present during `docker build`), producing wrong-arch kernels that run
# incorrectly on MI300X — e.g. gpu-copy-bw:correctness fails its CheckBuf data check.
# Override at build time with: --build-arg AMDGPU_TARGETS="gfx90a gfx942 gfx950".
ARG AMDGPU_TARGETS="gfx942"
ENV AMDGPU_TARGETS="${AMDGPU_TARGETS}"

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
    rm -rf /tmp/ompi

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/866182/mlc_v3.12.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Install RCCL
RUN cd /opt/ && \
    git clone -b release/rocm-rel-6.4 https://github.com/ROCmSoftwarePlatform/rccl.git && \
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
# rocm_megatron_lm: skipped (broken upstream - pretrain_deepseek.py missing in rocm_dev branch).
# apex_rocm: skipped - all apex imports in Megatron-DeepSpeed are guarded with try/except,
#   superbench has zero direct apex usage, and PyTorch 2.7 has native fused optimizers/AMP.
RUN make RCCL_HOME=/opt/rccl/build/ ROCBLAS_BRANCH=release-staging/rocm-rel-6.4 HIPBLASLT_BRANCH=release-staging/rocm-rel-6.4 ROCM_VER=rocm-5.5.0 -C third_party rocm -o cpu_hpl -o cpu_stream -o megatron_lm -o rocm_hipblaslt -o rocm_megatron_lm -o apex_rocm

# Build hipblaslt separately with the Tensile target-triple fix for the ROCm 6.4 clang.
# Also work around a joblib race (github.com/joblib/joblib/issues/1788, fixed upstream in
# PR #1789 but not yet in a joblib release) that crashes Tensile's parallel library build
# on Python 3.12 with "RuntimeError: Set changed size during iteration". The primary fix is
# applied below by patching Tensile to use the ordered generator (return_as="generator"),
# which avoids the buggy generator_unordered code path entirely. The system-wide joblib
# source patch here is kept as defense-in-depth for the base env; note it cannot reach the
# fresh --clear virtualenv that Tensile's install.sh creates and pip-installs joblib into.
RUN pip install "joblib>=1.4.2" && \
    joblib_parallel_py=$(python3 -c 'import joblib, pathlib; print(pathlib.Path(joblib.__file__).resolve().parent / "parallel.py")') && \
    sed -i 's/timeout_control_job = next(iter(self\._jobs_set), None)/timeout_control_job = next(iter(set(self._jobs_set)), None)/' "${joblib_parallel_py}"
RUN cd third_party && \
    git clone -b release-staging/rocm-rel-6.4 https://github.com/ROCmSoftwarePlatform/hipBLASLt.git && \
    sed -i 's/host-x86_64-unknown-linux,/host-x86_64-unknown-linux-gnu,/' \
        hipBLASLt/tensilelite/Tensile/BuildCommands/SharedCommands.py && \
    find hipBLASLt/tensilelite -type f -name '*.py' -exec sed -i -E \
        "s/return_as=(['\"])generator_unordered\1/return_as=\1generator\1/g" {} + && \
    sed -i -E 's/make -j(\$\(nproc\)|[0-9]+)/make -j'"${NUM_MAKE_JOBS}"'/g' hipBLASLt/install.sh && \
    cd hipBLASLt && ./install.sh -dc -j ${NUM_MAKE_JOBS} && \
    cp -v build/release/clients/staging/hipblaslt-bench /opt/superbench/bin/
RUN cp -r /opt/superbench/third_party/hipBLASLt/build/release/hipblaslt-install/lib/*  /opt/rocm/lib/ && \
    cp -r /opt/superbench/third_party/hipBLASLt/build/release/hipblaslt-install/include/*  /opt/rocm/include/
RUN cd third_party/Megatron/Megatron-DeepSpeed && \
    git apply ../megatron_deepspeed_rocm6.patch

# Install TransformerEngine - pin to 386bd316 (before NVFP4/hip_fp4.h which needs ROCm 7.0+).
# Disable CK fused attention (aiter submodule has gfx950-only code); aotriton stays enabled.
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && \
    git checkout 386bd316 && \
    git submodule update --init --recursive && \
    NVTE_FRAMEWORK=pytorch \
    NVTE_FUSED_ATTN_CK=0 \
    NVTE_ROCM_ARCH=gfx942 \
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
