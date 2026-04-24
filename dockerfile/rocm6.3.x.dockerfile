ARG BASE_IMAGE=rocm/pytorch-training:v25.6

FROM ${BASE_IMAGE}

# Base image: rocm/pytorch-training:v25.6
# Pre-installed by base image:
#   - Ubuntu: 22.04
#   - Python: 3.10
#   - ROCm: 6.3.4
#   - openmpi: 4.0.7rc2
#   - torch: 2.8.0a0+git7d205b2
#   - rccl: 2.21.5.60304-76
#   - hipblaslt: 0.15.0-8c69191d
#   - transformer_engine: 1.14.0+2f85f5f2
#   - flash_attention: 3.0.0.post1
#   - cmake: 3.18.5
#   - rocm-cmake: 0.14.0.60304-76
#   - amd-smi: 25.1.0+8dc45db
# Added by this Dockerfile:
#   - Docker Client: 27.5.1
#   - mlc: v3.12
#   - OFED: 24.10-1.1.4.0 LTS

# Fix base image botocore/urllib3 incompatibility:
# Base image ships botocore 1.22.12 (expects urllib3 1.x) with urllib3 2.6.3,
# causing "cannot import name 'DEFAULT_CIPHERS' from 'urllib3.util.ssl_'".
# Upgrade botocore/boto3 to specific versions compatible with urllib3 2.x.
RUN python3 -m pip install --no-cache-dir "botocore==1.35.98" "boto3==1.35.98"

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

# Install Docker
ENV DOCKER_VERSION=27.5.1
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


# Install OFED
ENV OFED_VERSION=24.10-1.1.4.0
# Check if ofed_info command is present; install OFED if missing
RUN if ! command -v ofed_info >/dev/null 2>&1; then \
    echo "OFED not found. Installing OFED..."; \
    UBUNTU_VERSION=$(lsb_release -r -s); \
    cd /tmp && \
    wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
    PATH=/usr/bin:${PATH} MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu${UBUNTU_VERSION}-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf MLNX_OFED_LINUX-${OFED_VERSION}* ; \
    fi

ENV ROCM_PATH=/opt/rocm

# Target GPU architectures for ROCm builds (space-separated)
ENV AMDGPU_TARGETS="gfx908 gfx90a gfx942"

# Use pre-installed OpenMPI from base image at /opt/ompi
ENV MPI_HOME=/opt/ompi

# Install Intel MLC
RUN cd /tmp && \
    wget -q https://downloadmirror.intel.com/866182/mlc_v3.12.tgz -O mlc.tgz && \
    tar xzf mlc.tgz Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/ && \
    rm -rf ./Linux mlc.tgz

ENV PATH="/opt/ompi/bin:/opt/superbench/bin:/usr/local/bin/:/opt/rocm/hip/bin/:/opt/rocm/bin/:${PATH}" \
    LD_LIBRARY_PATH="/opt/ompi/lib:/usr/lib/x86_64-linux-gnu/:/usr/local/lib/:/opt/rocm/lib:${LD_LIBRARY_PATH}" \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN echo PATH="$PATH" > /etc/environment && \
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >> /etc/environment && \
    echo SB_MICRO_PATH="$SB_MICRO_PATH" >> /etc/environment

RUN python3 -m pip install --upgrade pip wheel setuptools==65.7

RUN python3 -c "import pkg_resources" || python3 -m pip install setuptools

WORKDIR ${SB_HOME}

ADD third_party third_party

RUN make RCCL_HOME=/opt/rocm ROCBLAS_BRANCH=release-staging/rocm-rel-6.3 HIPBLASLT_BRANCH=release-staging/rocm-rel-6.3 ROCM_VER=rocm-5.5.0 -C third_party rocm -o cpu_hpl -o cpu_stream -o megatron_lm -o rocm_megatron_lm

ADD . .
ENV USE_HIP_DATATYPE=1
ENV USE_HIPBLAS_COMPUTETYPE=1
RUN python3 -m pip install --no-build-isolation .[amdworker]  && \
    CXX=/opt/rocm/bin/hipcc make cppbuild  && \
    make postinstall && \
    rm -rf .git
