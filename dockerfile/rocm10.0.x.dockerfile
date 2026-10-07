ARG BASE_IMAGE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c
FROM ${BASE_IMAGE}

LABEL maintainer="SuperBench"
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Experimental scaffold: preserve AMD's matched Python 3.14 / PyTorch 2.13 stack.
# SDK packages include all devices; this argument only selects locally built kernels.
ARG AMDGPU_TARGETS="gfx942 gfx950 gfx1250"
ARG NUM_MAKE_JOBS=64
ARG ROCM_VERSION=10.0.0
# rocm-systems revision selected by TheRock's therock-10.0 release.
ARG ROCM_SYSTEMS_COMMIT=6b0e43f341195e203754e08f850e437ff2fc09f9

ENV DEBIAN_FRONTEND=noninteractive \
    AMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${NUM_MAKE_JOBS}" \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    MPI_HOME=/usr/local/mpi \
    EXTRAS_PATH=/opt/rocm/extras-10 \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf automake bc build-essential ca-certificates cmake curl \
        dmidecode flex gfortran git gnupg iproute2 jq libaio-dev \
        libatomic1 libboost-program-options-dev libcap2 libcurl4-openssl-dev \
        libibumad-dev libibverbs-dev libnuma-dev libpci-dev libquadmath0 librdmacm-dev \
        libssl-dev libtinfo6 libtool lshw net-tools ninja-build numactl \
        openssh-client openssh-server pciutils pkg-config python3-dev rsync sudo \
        util-linux wget && \
    apt-get clean

# Install the Docker CLI client only.
ARG DOCKER_VERSION=29.6.2
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" -o /tmp/docker.tgz && \
    tar --extract --file /tmp/docker.tgz --strip-components 1 --directory /usr/local/bin/ docker/docker && \
    rm /tmp/docker.tgz

# Update system config.
RUN mkdir -p /root/.ssh /var/run/sshd && \
    touch /root/.ssh/authorized_keys && \
    sed -i 's/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/; s/[# ]*PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config && \
    printf '%s\n' '* soft nofile 1048576' '* hard nofile 1048576' >> /etc/security/limits.conf

# Match the Ubuntu 26.04 Azure HPC userspace stack; never install host drivers here.
# TODO: Add a selectable alternative networking stack after its qualification.
ARG DOCA_VERSION=3.4.0
ARG DOCA_PACKAGE_VERSION=3.4.0-085000
ARG DOCA_OPENMPI_VERSION=5.0.10rc2.2605121430-1.b99be7132e
ARG DOCA_BUNDLE_SHA256=60789bbbfc9b283a87a64589a0d4bd5d08d1b746494c24ce05e10bbd694de6bb
# Use NVIDIA's signed local repository bundle: the online repository's TLS EOF
# handling fails with Ubuntu 26.04 APT. Do not disable certificate/signature checks.
RUN curl -fsSL "https://content.mellanox.com/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host_${DOCA_PACKAGE_VERSION}-26.04-ubuntu2604_amd64.deb" -o /tmp/doca-host.deb && \
    printf '%s  %s\n' "${DOCA_BUNDLE_SHA256}" /tmp/doca-host.deb | sha256sum --check --strict && \
    dpkg -i /tmp/doca-host.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        "doca-ofed-userspace=${DOCA_PACKAGE_VERSION}" "openmpi=${DOCA_OPENMPI_VERSION}" && \
    mpi_prefix=$(dpkg-query -L openmpi | sed -n 's#/bin/mpirun$##p') && \
    test -n "${mpi_prefix}" && \
    test -x "${mpi_prefix}/bin/mpicc" && \
    ln -s "${mpi_prefix}" "${MPI_HOME}" && \
    printf '%s\n' "${MPI_HOME}/lib" > /etc/ld.so.conf.d/superbench-mpi.conf && \
    ldconfig && \
    "${MPI_HOME}/bin/mpirun" --version && \
    apt-get purge -y doca-host && \
    rm /tmp/doca-host.deb && \
    apt-get update && apt-get clean

# Add development files to the base's SDK and use AMD's path discovery directly.
# RCCL is AMD's prebuilt library in rocm-sdk-libraries, never built from source.
RUN python3 -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
        "rocm[devel,libraries,device-all]==${ROCM_VERSION}" && \
    rocm-sdk init && \
    ROCM_PATH="$(rocm-sdk path --root)" && \
    test -r "${ROCM_PATH}/.info/version" && \
    test -r "${ROCM_PATH}/lib/librccl.so" && \
    printf 'export ROCM_PATH=%q\n' "${ROCM_PATH}" > /etc/profile.d/rocm10-env.sh && \
    printf '%s\n' \
        'export ROCM_HOME="${ROCM_PATH}" HIP_PATH="${ROCM_PATH}" RCCL_HOME="${ROCM_PATH}"' \
        'export PATH="${EXTRAS_PATH}/bin:${MPI_HOME}/bin:${SB_MICRO_PATH}/bin:${ROCM_PATH}/bin:${PATH}"' \
        'export LD_LIBRARY_PATH="${EXTRAS_PATH}/lib:${MPI_HOME}/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib:${ROCM_PATH}/lib/host-math/lib:${ROCM_PATH}/lib/rocm_sysdeps/lib"' \
        >> /etc/profile.d/rocm10-env.sh

# Resolve the SDK once at build time; shell startup must not reinitialize it.
ENV BASH_ENV=/etc/profile.d/rocm10-env.sh
ENTRYPOINT ["/bin/bash", "-e", "-c", "exec \"$@\"", "--"]
CMD ["/bin/bash"]

# Register the bundled module until https://github.com/ROCm/TheRock/issues/3054
# is fixed (upstream registration: https://github.com/ROCm/TheRock/pull/8424).
# A relative .pth entry preserves the wrapper's lookup of its matching library.
RUN python3 -c 'import sysconfig; from pathlib import Path; Path(sysconfig.get_path("purelib"), "amdsmi.pth").write_text("_rocm_sdk_core/share/amd_smi\n")' && \
    python3 -c 'import amdsmi'

# Built-in MPI calls exchange host data; RCCL exchanges GPU buffers. The DOCA
# Open MPI package needs neither a ROCm-aware rebuild nor unused OpenSHMEM.
ENV SB_ROCM_MEM_BW_BACKEND=transferbench \
    USE_HIP_DATATYPE=1 \
    USE_HIPBLAS_COMPUTETYPE=1

# Install RVS through AMD's signed Ubuntu 26.04 repositories. Explicitly select
# matching native dependencies instead of legacy distro packages or ROCm 10.1.
ARG RVS_VERSION=1.6.122-708
ARG ROCM_APT_VERSION=10.0.0-4
RUN install -d -m 0755 /etc/apt/keyrings && \
    curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg | \
        gpg --batch --yes --dearmor -o /etc/apt/keyrings/amdrocm.gpg && \
    printf '%s\n' 'Types: deb' \
        'URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/' \
        'Suites: stable' 'Components: main' 'Architectures: amd64' \
        'Signed-By: /etc/apt/keyrings/amdrocm.gpg' > /etc/apt/sources.list.d/amdrocm-core.sources && \
    printf '%s\n' 'Types: deb' \
        'URIs: https://stable.repo.amd.com/rocm/extras/rvs/packages/ubuntu2604/' \
        'Suites: stable' 'Components: main' 'Architectures: amd64' \
        'Signed-By: /etc/apt/keyrings/amdrocm.gpg' > /etc/apt/sources.list.d/amdrocm-rvs.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        "amdrocm-runtime=${ROCM_APT_VERSION}" "amdrocm-base=${ROCM_APT_VERSION}" \
        "amdrocm-amdsmi=${ROCM_APT_VERSION}" "amdrocm-blas=${ROCM_APT_VERSION}" \
        "amdrocm-rand=${ROCM_APT_VERSION}" "amdrocm-llvm=${ROCM_APT_VERSION}" \
        "amdrocm10-rvs=${RVS_VERSION}" && \
    apt-get clean && \
    test -x "${EXTRAS_PATH}/bin/rvs" && \
    test -x "${EXTRAS_PATH}/bin/TransferBench"

WORKDIR ${SB_HOME}
COPY third_party third_party

# Build only the release-matched RCCL test clients against the packaged library.
RUN git init /tmp/rocm-systems && \
    git -C /tmp/rocm-systems remote add origin https://github.com/ROCm/rocm-systems.git && \
    git -C /tmp/rocm-systems sparse-checkout set projects/rccl-tests && \
    git -C /tmp/rocm-systems fetch --depth 1 --filter=blob:none origin "${ROCM_SYSTEMS_COMMIT}" && \
    git -C /tmp/rocm-systems checkout --detach FETCH_HEAD && \
    rccl_architectures=$(printf '%s' "${AMDGPU_TARGETS}" | tr -s '[:space:]' ';' | sed 's/^;//; s/;$//') && \
    if [ -z "${rccl_architectures}" ]; then \
        echo 'AMDGPU_TARGETS must contain at least one GPU architecture.' >&2; exit 1; \
    fi && \
    cmake -S /tmp/rocm-systems/projects/rccl-tests -B /tmp/rccl-tests-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH};${MPI_HOME}" \
        -DROCM_PATH="${ROCM_PATH}" \
        -DGPU_TARGETS="${rccl_architectures}" \
        -DUSE_MPI=ON \
        -DCMAKE_INSTALL_PREFIX="${SB_HOME}" \
        -DCMAKE_INSTALL_RPATH="${ROCM_PATH}/lib;${MPI_HOME}/lib" \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON && \
    cmake --build /tmp/rccl-tests-build -j"${NUM_MAKE_JOBS}" && \
    cmake --install /tmp/rccl-tests-build && \
    rm -rf /tmp/rocm-systems /tmp/rccl-tests-build

# Do not invoke the legacy ROCm aggregate target: it clones HIP 5.5 samples,
# rebuilds math libraries, and installs unqualified Megatron/DeepSpeed dependencies.
RUN test -f third_party/fio/Makefile && \
    test -f third_party/perftest/autogen.sh && \
    make -C third_party fio rocm_perftest

# Extract only matching prebuilt clients and their external GPU kernel packs.
# The ELFs alone cannot initialize matrices without the blas_test_*.kpack files.
ARG ROCM_TESTS_SHA256=47468ff71b8027eed4784de5052aa18684cb2ec0bfe0409ad7e62b9a6cf88f1f
RUN curl -fsSL "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-multiarch-tests-${ROCM_VERSION}.tar.gz" -o /tmp/rocm-tests.tar.gz && \
    printf '%s  %s\n' "${ROCM_TESTS_SHA256}" /tmp/rocm-tests.tar.gz | sha256sum --check --strict && \
    tar -xzf /tmp/rocm-tests.tar.gz -C "${SB_HOME}" --wildcards --no-wildcards-match-slash \
        './bin/rocblas-bench' './bin/hipblaslt-bench' './.kpack/blas_test_*.kpack' && \
    rm /tmp/rocm-tests.tar.gz && \
    test -x "${SB_HOME}/bin/rocblas-bench" && \
    test -x "${SB_HOME}/bin/hipblaslt-bench" && \
    test -s "${SB_HOME}/.kpack/blas_test_gfx942.kpack" && \
    test -s "${SB_HOME}/.kpack/blas_test_gfx950.kpack" && \
    test -s "${SB_HOME}/.kpack/blas_test_gfx1250.kpack"

COPY . .
# Keep AMD's framework and all-device SDK packages when resolving SuperBench deps.
# Resolve these before the framework installer applies its tested dependency pins.
# The SDK supplies amdsmi via .pth, not a separate distribution; use the torch
# extra so amdworker does not install a PyPI wrapper that shadows the SDK module.
RUN python3 -m pip freeze | grep -Ei '^(torch|torchvision|torchaudio|triton|rocm|rocm-sdk-[^=]+)==' > /tmp/rocm10-constraints.txt && \
    python3 -m pip install -c /tmp/rocm10-constraints.txt '.[torch]' && \
    python3 -m pip install pytest pytest-timeout vcrpy

# Install TransformerEngine, AITER and Megatron before the final native build.
ENV SB_ROCM10_FRAMEWORK_ROOT=/opt/rocm10-frameworks \
    SB_MEGATRON_PATH=/opt/rocm10-frameworks/Megatron-LM \
    AITER_USE_SYSTEM_TRITON=1 \
    TRITON_F32_DEFAULT=ieee \
    MAX_JOBS=16 \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NVTE_FLASH_ATTN=0 \
    NVTE_FUSED_ATTN=1 \
    NVTE_FUSED_ATTN_CK=1 \
    NVTE_FUSED_ATTN_AOTRITON=0
RUN framework_jobs="${NUM_MAKE_JOBS}" && \
    if [ "${framework_jobs}" -gt 16 ]; then framework_jobs=16; fi && \
    MAX_JOBS="${framework_jobs}" bash dockerfile/etc/install-rocm10-frameworks.sh && \
    printf '%s\n' \
        "PATH=${PATH}" \
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
        "SB_MICRO_PATH=${SB_MICRO_PATH}" \
        "SB_ROCM_MEM_BW_BACKEND=${SB_ROCM_MEM_BW_BACKEND}" \
        "ROCM_PATH=${ROCM_PATH}" \
        "ROCM_HOME=${ROCM_HOME}" \
        "HIP_PATH=${HIP_PATH}" \
        "RCCL_HOME=${RCCL_HOME}" \
        "SB_ROCM10_FRAMEWORK_ROOT=${SB_ROCM10_FRAMEWORK_ROOT}" \
        "SB_MEGATRON_PATH=${SB_MEGATRON_PATH}" \
        "AITER_USE_SYSTEM_TRITON=${AITER_USE_SYSTEM_TRITON}" \
        "TRITON_F32_DEFAULT=${TRITON_F32_DEFAULT}" \
        "MAX_JOBS=${MAX_JOBS}" \
        "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}" \
        "NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN}" \
        "NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN}" \
        "NVTE_FUSED_ATTN_CK=${NVTE_FUSED_ATTN_CK}" \
        "NVTE_FUSED_ATTN_AOTRITON=${NVTE_FUSED_ATTN_AOTRITON}" \
        > /etc/environment

# Build SuperBench and finish installation.
RUN CXX="${ROCM_PATH}/bin/hipcc" make cppbuild && \
    make postinstall

RUN python3 -m pip check && \
    sb --help && \
    rm /tmp/rocm10-constraints.txt && \
    rm -rf .git
