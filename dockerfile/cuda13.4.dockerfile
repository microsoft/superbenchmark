FROM nvcr.io/nvidia/pytorch:26.09-py3

# OS:
#   - Ubuntu: 24.04
#   - OpenMPI: 5.0.10rc2 (from HPC-X 2.51)
#   - Docker Client: 20.10.8 (installed in this dockerfile)
# NVIDIA:
#   - CUDA: 13.4 V13.4.59 (from pytorch:26.09-py3)
#   - cuDNN: 9.26.0
#   - cuBLAS: 13.8.0.4
#   - NCCL: 2.31.2
#   - TransformerEngine: 2.19.0
#   - torch: 2.14.0a0+b2c75dd062.nv26.09
#   - sm_107 / compute capability 10.7
# Mellanox (from base image — not reinstalled):
#   - OFED: inbox (kernel-provided)
#   - HPC-X: 2.51 (includes ompi4 + ompi5, UCX 1.22.0)
# Intel:
#   - mlc: 3.12 (amd64 only)
#
# Notes for sm_107:
#   - This machine is aarch64 (ARM). cpu_hpl, Intel MLC, AOCC and AMD BLIS auto-skip on aarch64.
#   - CUDA 13.4 builds the in-tree CUDA benchmarks, CUTLASS, NVBench,
#     cuBLASLt and NCCL tests for sm_107; gpu-burn generates compute_107 PTX.
#
# Build (from repo root), e.g.:
#   docker build -t superbench-cuda13.4 \
#     --build-arg NUM_MAKE_JOBS=64 \
#     -f dockerfile/cuda13.4.dockerfile .

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

# Upgrade GCC to 15 and binutils to 2.46.1 — required for -mcpu=olympus support
# (Ubuntu 24.04 ships GCC 13 / binutils 2.42 which lack Olympus CPU definitions).
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-15 g++-15 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 150 \
    --slave   /usr/bin/g++  g++  /usr/bin/g++-15 \
    --slave   /usr/bin/gcov gcov /usr/bin/gcov-15 && \
    update-alternatives --install /usr/bin/cc  cc  /usr/bin/gcc-15 150 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-15 150 && \
    update-alternatives --set gcc /usr/bin/gcc-15 && \
    update-alternatives --set cc  /usr/bin/gcc-15 && \
    update-alternatives --set c++ /usr/bin/g++-15 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ENV BINUTILS_VERSION=2.46.1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget ca-certificates xz-utils zlib1g-dev libzstd-dev && \
    cd /tmp && \
    wget -nv https://ftp.gnu.org/gnu/binutils/binutils-${BINUTILS_VERSION}.tar.xz && \
    tar -xf binutils-${BINUTILS_VERSION}.tar.xz && \
    cd binutils-${BINUTILS_VERSION} && \
    ./configure --prefix=/usr/local \
    --enable-plugins \
    --enable-64-bit-bfd \
    --with-system-zlib \
    --disable-werror && \
    make -j"$(nproc)" MAKEINFO=true && \
    make install MAKEINFO=true && \
    for t in as ld ld.bfd nm ar ranlib objcopy objdump strip readelf addr2line c++filt size strings gprof; do \
    if [ -x /usr/local/bin/$t ]; then ln -sf /usr/local/bin/$t /usr/bin/$t; fi; \
    done && \
    ln -sf /usr/local/bin/as /usr/bin/aarch64-linux-gnu-as && \
    ln -sf /usr/local/bin/ld /usr/bin/aarch64-linux-gnu-ld && \
    cd / && \
    rm -rf /tmp/binutils-${BINUTILS_VERSION}* && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ENV PATH="/usr/local/bin:${PATH}"

# Machine has 176 cores (352 threads); raise make jobs to speed up the build.
# Max ~176 (cores) / 352 (threads). Override at build time with --build-arg NUM_MAKE_JOBS=...
ARG NUM_MAKE_JOBS=64
ARG TARGETPLATFORM
ARG TARGETARCH

ENV CUDA_ARCH_LIST="10.7"
ENV TORCH_CUDA_ARCH_LIST="10.7"

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

# OFED and HPC-X: Using the base image's versions (inbox OFED, HPC-X 2.51).
# The base NGC pytorch:26.09-py3 image ships HPC-X at /opt/hpcx with
# ompi4+ompi5 and the ompi_mpi_short_float symbol that PyTorch is linked against.
# Note: HPC-X 2.51 no longer has hpcx-init.sh; use /opt/hpcx/ompi/bin directly.
# DO NOT install a separate OFED or HPC-X — it breaks PyTorch's MPI linkage.
# The commented-out sections below are kept for reference only.
#
# ENV OFED_VERSION=24.10-1.1.4.0
# RUN TARGETARCH_HW=$(uname -m) && \
#     cd /tmp && \
#     wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu24.04-${TARGETARCH_HW}.tgz && \
#     tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu24.04-${TARGETARCH_HW}.tgz && \
#     MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu24.04-${TARGETARCH_HW}/mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force --all && \
#     rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*
#
# ENV HPCX_VERSION=v2.24.1
# RUN TARGETARCH_HW=$(uname -m) && \
#     cd /opt && \
#     rm -rf hpcx && \
#     wget https://content.mellanox.com/hpc/hpc-x/${HPCX_VERSION}_cuda13/hpcx-${HPCX_VERSION}-gcc-doca_ofed-ubuntu24.04-cuda13-${TARGETARCH_HW}.tbz -O hpcx.tbz && \
#     tar xf hpcx.tbz && \
#     mv hpcx-${HPCX_VERSION}-gcc-doca_ofed-ubuntu24.04-cuda13-${TARGETARCH_HW} hpcx && \
#     rm hpcx.tbz

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
    echo "Skipping Intel MLC, AOCC and AMD BLIS installations for non-amd64 architecture: $TARGETARCH"; \
    fi

# Install UCX with multi-threading support
# Note: UCX 1.18.0 is incompatible with GCC 15 (omp.h templates inside extern "C",
# plus incompatible-pointer-type casts). Build with gcc-13 which is still installed.
ENV UCX_VERSION=1.18.0
RUN cd /tmp && \
    wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}-rc1/ucx-${UCX_VERSION}.tar.gz && \
    tar xzf ucx-${UCX_VERSION}.tar.gz && \
    cd ucx-${UCX_VERSION} && \
    CC=gcc-13 CXX=g++-13 ./contrib/configure-release-mt --prefix=/usr/local && \
    make -j ${NUM_MAKE_JOBS} && \
    make install

# Add the base image's HPC-X 2.51 ompi to PATH so mpicc is available for builds.
ENV MPI_HOME=/opt/hpcx/ompi
ENV PATH="/opt/hpcx/ompi/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment

# Add config files
ADD dockerfile/etc /opt/microsoft/

WORKDIR ${SB_HOME}

ADD third_party third_party
# Build all CUDA targets. The base image's HPC-X 2.51 provides mpicc on PATH
# (via MPI_HOME=/opt/hpcx/ompi set above).
RUN make -C third_party cuda NUM_MAKE_JOBS=${NUM_MAKE_JOBS}

ADD . .
RUN python3 -m pip install --upgrade setuptools==78.1.0 && \
    python3 -m pip install --no-cache-dir .[nvworker] && \
    make cppbuild && \
    make postinstall && \
    rm -rf .git
