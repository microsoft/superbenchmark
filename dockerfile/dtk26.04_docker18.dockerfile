ARG BASE_IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/vllm:0.11.0-ubuntu22.04-dtk26.04-py3.10

FROM ${BASE_IMAGE}

# OS:
#   - Ubuntu: 22.04
#   - Docker Client: 20.10.8
# DTK:
#   - DTK: 26.04
# Lib:
#   - ucx: 1.20.0
#   - openmpi: 5.0.9
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
    python3.10-venv \
    rsync \
    sudo \
    util-linux \
    vim \
    wget \
    && \
    rm -rf /tmp/*

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

ENV ROCM_PATH=/opt/dtk

# Docker 18.09 legacy builder cannot use BuildKit-only named contexts or
# RUN --mount. Prepare a local ./hyhal directory in the build context before
# running docker build, then copy it into the image.
COPY hyhal /opt/hyhal

# Install UCX
ARG UCX_VERSION=1.20.0
ARG UCX_HOME=/opt/ucx
RUN cd /tmp && \
    wget https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz && \
    tar xzf ucx-${UCX_VERSION}.tar.gz && \
    cd ucx-${UCX_VERSION} && \
    ./contrib/configure-release --prefix=${UCX_HOME} \
    --enable-optimizations --enable-tuning \
    --enable-cma --enable-mt \
    --with-mlx5 --with-rc --with-ud --with-dc --with-dm --with-ib_hw_tm \
    --with-verbs=/usr/include --with-rdmacm=/usr \
    --with-rocm=${ROCM_PATH} \
    --without-knem --without-cuda --without-java && \
    make -j $(nproc) && \
    rm -rf ${UCX_HOME} && \
    make install && \
    rm -rf /tmp/ucx-${UCX_VERSION}*

# Install OpenMPI
ENV MPI_HOME=/opt/mpi
ARG OMPI_VERSION=5.0.9
RUN cd /tmp && \
    wget https://download.open-mpi.org/release/open-mpi/v${OMPI_VERSION%.*}/openmpi-${OMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OMPI_VERSION}.tar.gz && \
    cd openmpi-${OMPI_VERSION} && \
    ./configure --prefix=${MPI_HOME} \
    --with-ucx=${UCX_HOME} \
    --with-rocm=${ROCM_PATH} \
    --enable-builtin-atomics \
    --enable-wrapper-rpath \
    --enable-mca-no-build=btl-uct \
    --enable-prte-prefix-by-default && \
    make -j $(nproc) && \
    rm -rf ${MPI_HOME} && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/openmpi-${OMPI_VERSION}*

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/866182/mlc_v3.12.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

# Add rocblas-bench to path
RUN ln -s ${ROCM_PATH}/lib/rocblas/benchmark_tool/rocblas-bench ${ROCM_PATH}/bin/ && \
    chmod +x ${ROCM_PATH}/bin/rocblas-bench && \
    ln -s ${ROCM_PATH}/lib/hipblaslt/benchmark_tool/hipblaslt-bench ${ROCM_PATH}/bin/ && \
    chmod +x ${ROCM_PATH}/bin/hipblaslt-bench

ENV PATH="${MPI_HOME}/bin:${UCX_HOME}/bin:/opt/superbench/bin:/usr/local/bin/${PATH:+:${PATH}}" \
    LD_LIBRARY_PATH="${MPI_HOME}/lib:${UCX_HOME}/lib:/usr/lib/x86_64-linux-gnu/:/usr/local/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

WORKDIR ${SB_HOME}

COPY third_party third_party
COPY dockerfile/etc/dtk26.04-topo-mapping.xml ${ROCM_PATH}/rccl/lib/topo_mapping_default.xml

RUN make \
    RCCL_HOME=${ROCM_PATH}/rccl \
    ROCM_PATH=${ROCM_PATH} \
    HIP_HOME=${ROCM_PATH}/hip \
    MPI_HOME=${MPI_HOME} \
    -C third_party \
    dtk \
    -o cpu_hpl \
    -o cpu_stream \
    -o megatron_lm \
    -o apex_rocm \
    -o megatron_deepspeed \
    -o rocm_megatron_lm

COPY . .

ARG SB_PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ENV USE_HIP_DATATYPE=1 \
    USE_HIPBLAS_COMPUTETYPE=1 \
    VIRTUAL_ENV=/opt/superbench-venv
ENV PATH="${VIRTUAL_ENV}/bin:${MPI_HOME}/bin:${UCX_HOME}/bin:/opt/superbench/bin:/usr/local/bin/${PATH:+:${PATH}}"

RUN sed -i '/NCCL_/d' /etc/bash.bashrc && \
    echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment && \
    echo VIRTUAL_ENV="$VIRTUAL_ENV" >> /etc/environment

RUN python3 -m venv --system-site-packages ${VIRTUAL_ENV} && \
    python3 -m pip install -i ${SB_PIP_INDEX_URL} --upgrade pip wheel setuptools==65.7 mpi4py && \
    python3 -m pip install -i ${SB_PIP_INDEX_URL} --no-build-isolation .[hgworker] && \
    make cppbuild  && \
    make postinstall
