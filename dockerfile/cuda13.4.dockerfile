# Build the streamOlympus STREAM variant in a separate stage. -mcpu=olympus needs GCC 15 and
# binutils 2.46, while the base image ships GCC 13 / binutils 2.42. Building it here keeps
# that toolchain (and its runtime libraries) out of the final image.
FROM nvcr.io/nvidia/pytorch:26.09-py3 AS stream-olympus-builder

ARG NUM_MAKE_JOBS=64
ENV DEBIAN_FRONTEND=noninteractive \
    BINUTILS_VERSION=2.46.1 \
    BINUTILS_SHA256=e127a709cba24c76de8936cb7083dd768f28cd37eb010492e2f19b71eb1294e4

COPY third_party/stream-tests/Makefile /build/Makefile

RUN mkdir -p /out && \
    if [ "$(uname -m)" != "aarch64" ]; then \
    echo "Skipping streamOlympus for non-aarch64 architecture"; \
    exit 0; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates wget xz-utils zlib1g-dev libzstd-dev && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-15 && \
    cd /tmp && \
    for base in https://sourceware.org/pub/binutils/releases https://mirrors.kernel.org/gnu/binutils https://ftp.gnu.org/gnu/binutils; do \
    wget -nv --tries=3 --timeout=60 "${base}/binutils-${BINUTILS_VERSION}.tar.xz" -O binutils.tar.xz && break; \
    done && \
    echo "${BINUTILS_SHA256}  binutils.tar.xz" | sha256sum -c - && \
    tar -xf binutils.tar.xz && \
    cd binutils-${BINUTILS_VERSION} && \
    ./configure --prefix=/opt/binutils-${BINUTILS_VERSION} \
    --with-system-zlib \
    --disable-werror \
    --disable-gprofng && \
    make -j "${NUM_MAKE_JOBS}" MAKEINFO=true && \
    make install MAKEINFO=true && \
    cd /build && \
    wget -nv https://www.cs.virginia.edu/stream/FTP/Code/stream.c && \
    make OLYMPUS CC="gcc-15 -B/opt/binutils-${BINUTILS_VERSION}/bin/" && \
    cp streamOlympus /out/

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
#   - mlc: 3.13 (amd64 only)
#
# Notes for sm_107:
#   - This machine is aarch64 (ARM). cpu_hpl, Intel MLC, AOCC and AMD BLIS auto-skip on aarch64.
#   - CUDA 13.4 builds the in-tree CUDA benchmarks, CUTLASS, NVBench,
#     cuBLASLt and NCCL tests for sm_107.
#   - streamOlympus is built with GCC 15 in the stream-olympus-builder stage above.
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

# Number of parallel jobs for the NVBench build. Override with --build-arg NUM_MAKE_JOBS=...
ARG NUM_MAKE_JOBS=64

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

# OFED and HPC-X: Using the base image's versions (inbox OFED, HPC-X 2.51).
# The base NGC pytorch:26.09-py3 image ships HPC-X at /opt/hpcx with
# ompi4+ompi5 and the ompi_mpi_short_float symbol that PyTorch is linked against.
# Note: HPC-X 2.51 no longer has hpcx-init.sh; use /opt/hpcx/ompi/bin directly.
# DO NOT install a separate OFED or HPC-X — it breaks PyTorch's MPI linkage.

# Installs specific to amd64 platform
RUN if [ "$TARGETARCH" = "amd64" ]; then \
    # Install Intel MLC
    cd /tmp && \
    wget -q https://downloadmirror.intel.com/926327/mlc_v3.13.tgz -O mlc.tgz && \
    echo "a8537e8ff3fad626d75a383fabc224ccc4cc98a0111c9989f7fb26b639f12019  mlc.tgz" | sha256sum -c - && \
    tar xzf mlc.tgz Linux/mlc && \
    install -m 755 ./Linux/mlc /usr/local/bin/ && \
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

# Add the base image's HPC-X 2.51 UCX and ompi to the library path and add mpicc to PATH.
ENV MPI_HOME=/opt/hpcx/ompi
ENV PATH="/opt/hpcx/ompi/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}" \
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
# Build the CUDA targets (the `cuda` aggregate; MSCCL is not included). mpicc comes from the
# base image's HPC-X 2.51 via the PATH set above.
RUN make -C third_party cuda NUM_MAKE_JOBS=${NUM_MAKE_JOBS}
COPY --from=stream-olympus-builder /out/ ${SB_MICRO_PATH}/bin/

ADD . .
RUN python3 -m pip install --upgrade setuptools==78.1.0 && \
    python3 -m pip install --no-cache-dir .[nvworker] && \
    make cppbuild && \
    make postinstall && \
    rm -rf .git
