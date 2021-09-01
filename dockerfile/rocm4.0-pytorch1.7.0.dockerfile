FROM rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch_1.7.0

# OS:
#   - Ubuntu: 18.04
#   - OpenMPI: 4.0.0
# AMD:
#   - ROCm: 4.0
#   - HIP: 3.21.2
#   - MIOpen: 2.9.0
#   - RCCL: 2.7.8
# Mellanox:
#   - OFED: 5.2-2.2.3.0

LABEL maintainer="SuperBench"

ENV DEBIAN_FRONTEND=noninteractive
RUN wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key add - && \
    apt-get update && \
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

# Install OpenMPI
ENV OPENMPI_VERSION=4.0.0
RUN cd /tmp && \
    wget -q https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf openmpi-${OPENMPI_VERSION}*

# Configure SSH
RUN mkdir -p /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i "s/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*Port.*/Port 22/" /etc/ssh/sshd_config && \
    echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config

# Install OFED
ENV OFED_VERSION=5.2-2.2.3.0
RUN cd /tmp && \
    wget -q http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tgz && \
    PATH=/usr/bin:${PATH} MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf MLNX_OFED_LINUX-${OFED_VERSION}*

ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" \
    SB_HOME="/opt/superbench" \
    SB_MICRO_PATH="/opt/superbench"

WORKDIR ${SB_HOME}

ADD third_party third_party
RUN ROCM_VERSION=rocm-4.0.0 make -j -C third_party rocm

# Workaround for image having package installed in user path
RUN mv /root/.local/bin/* /opt/conda/bin/ && \
    mv /root/.local/lib/python3.6/site-packages/* /opt/conda/lib/python3.6/ && \
    rm -rf /root/.local

ADD . .
RUN python3 -m pip install .[torch] && \
    make cppbuild
