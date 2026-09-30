ARG BASE_IMAGE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c
FROM ${BASE_IMAGE}

LABEL maintainer="SuperBench"
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]

# Experimental scaffold: preserve AMD's matched Python 3.14 / PyTorch 2.13 stack.
# SDK packages include all devices; this argument only selects locally built kernels.
ARG AMDGPU_TARGETS="gfx942 gfx950"
ARG NUM_MAKE_JOBS=8
ARG ROCM_VERSION=10.0.0
ARG OPENMPI_VERSION=4.1.8
# rocm-systems revision selected by TheRock's therock-10.0 release.
ARG ROCM_SYSTEMS_COMMIT=6b0e43f341195e203754e08f850e437ff2fc09f9

ENV DEBIAN_FRONTEND=noninteractive \
    AMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${NUM_MAKE_JOBS}" \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    MPI_HOME=/usr/local/mpi \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    RCCL_HOME=/opt/rocm \
    SB_HOME=/opt/superbench \
    SB_MICRO_PATH=/opt/superbench \
    ANSIBLE_DEPRECATION_WARNINGS=FALSE \
    ANSIBLE_COLLECTIONS_PATH=/usr/share/ansible/collections

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf automake bc build-essential ca-certificates cmake curl \
        dmidecode flex gfortran git iproute2 jq libaio-dev \
        libatomic1 libboost-program-options-dev libcap2 libcurl4-openssl-dev \
        libibumad-dev libibverbs-dev libnuma-dev libpci-dev libquadmath0 librdmacm-dev \
        libssl-dev libtinfo6 libtool lshw net-tools ninja-build numactl \
        openssh-client openssh-server pciutils pkg-config rsync sudo \
        util-linux wget && \
    apt-get clean

# The base uses ROCm wheels, not DEBs. Add development files to that same SDK;
# installing amdrocm-core-sdk as well would introduce a second runtime.
# RCCL is AMD's prebuilt library in rocm-sdk-libraries, never built from source.
RUN python3 -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
        "rocm[devel,libraries,device-all]==${ROCM_VERSION}" && \
    rocm-sdk init && \
    ln -s "$(rocm-sdk path --root)" "${ROCM_PATH}" && \
    test -r "${ROCM_PATH}/.info/version" && \
    test -r "${ROCM_PATH}/lib/librccl.so" && \
    printf '%s\n' "${ROCM_PATH}/lib" "${ROCM_PATH}/lib/llvm/lib" > /etc/ld.so.conf.d/rocm.conf && \
    ldconfig && \
    python3 -m pip install "${ROCM_PATH}/share/amd_smi"

# MPI in the built-in benchmarks exchanges host metadata/results; RCCL exchanges
# GPU buffers. Keep the existing MPI prefix without claiming HIP-aware MPI support.
RUN git clone --depth 1 --recursive -b "v${OPENMPI_VERSION}" https://github.com/open-mpi/ompi.git /tmp/ompi && \
    cd /tmp/ompi && \
    ./autogen.pl && \
    mkdir build && cd build && \
    ../configure --prefix="${MPI_HOME}" \
        --enable-orterun-prefix-by-default --enable-mpirun-prefix-by-default && \
    make -j"${NUM_MAKE_JOBS}" && \
    make install && \
    printf '%s\n' "${MPI_HOME}/lib" > /etc/ld.so.conf.d/superbench-mpi.conf && \
    ldconfig && \
    cd / && rm -rf /tmp/ompi

ENV PATH="/usr/local/mpi/bin:/opt/superbench/bin:/opt/rocm/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/mpi/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib" \
    USE_HIP_DATATYPE=1 \
    USE_HIPBLAS_COMPUTETYPE=1

ARG DOCKER_VERSION=29.6.2
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_VERSION}.tgz" -o /tmp/docker.tgz && \
    tar --extract --file /tmp/docker.tgz --strip-components 1 --directory /usr/local/bin/ docker/docker && \
    rm /tmp/docker.tgz

RUN mkdir -p /root/.ssh /var/run/sshd && \
    touch /root/.ssh/authorized_keys && \
    sed -i 's/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/; s/[# ]*PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config && \
    printf '%s\n' '* soft nofile 1048576' '* hard nofile 1048576' >> /etc/security/limits.conf && \
    printf 'PATH=%s\nLD_LIBRARY_PATH=%s\nSB_MICRO_PATH=%s\n' \
        "${PATH}" "${LD_LIBRARY_PATH}" "${SB_MICRO_PATH}" > /etc/environment

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

WORKDIR ${SB_HOME}
COPY third_party third_party
# Do not invoke the legacy ROCm aggregate target: it clones HIP 5.5 samples,
# rebuilds math libraries, and installs unqualified Megatron/DeepSpeed dependencies.
RUN test -f third_party/fio/Makefile && \
    test -f third_party/perftest/autogen.sh && \
    make -C third_party fio rocm_perftest

COPY . .
# Keep AMD's framework and all-device SDK packages when resolving SuperBench deps.
RUN python3 -m pip freeze | grep -Ei '^(torch|torchvision|torchaudio|triton|rocm|rocm-sdk-[^=]+)==' > /tmp/rocm10-constraints.txt && \
    python3 -m pip install -c /tmp/rocm10-constraints.txt '.[amdworker]' && \
    CXX="${ROCM_PATH}/bin/hipcc" make cppbuild && \
    make postinstall && \
    python3 -m pip install pytest pytest-timeout vcrpy && \
    python3 -m pip check && \
    sb --help && \
    rm /tmp/rocm10-constraints.txt && \
    rm -rf .git

# The official relocatable Extras archive works with the wheel SDK without
# pulling a second ROCm installation through native-package dependencies.
ARG RVS_VERSION=1.6.122-708
ARG RVS_SHA256=4ea57ddd8442f0e271e4d7cc8806019f7b0b008e258b883a38dd951c0097516a
ENV EXTRAS_PATH=/opt/rocm-extras-10
ENV PATH="${EXTRAS_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="${EXTRAS_PATH}/lib:${LD_LIBRARY_PATH}"
RUN curl -fsSL "https://stable.repo.amd.com/rocm/extras/rvs/tarball/amdrocm10-rvs-${RVS_VERSION}-Linux.tar.gz" -o /tmp/rvs.tar.gz && \
    printf '%s  %s\n' "${RVS_SHA256}" /tmp/rvs.tar.gz | sha256sum --check --strict && \
    mkdir -p "${EXTRAS_PATH}" && \
    tar -xzf /tmp/rvs.tar.gz -C "${EXTRAS_PATH}" && \
    rm /tmp/rvs.tar.gz && \
    test -x "${EXTRAS_PATH}/bin/rvs" && \
    test -x "${EXTRAS_PATH}/bin/TransferBench" && \
    printf 'PATH=%s\nLD_LIBRARY_PATH=%s\nSB_MICRO_PATH=%s\n' \
        "${PATH}" "${LD_LIBRARY_PATH}" "${SB_MICRO_PATH}" > /etc/environment

# Loading the devel overlay's AMD SMI copy after PyTorch's core copy creates two
# initialization states and reports zero GPUs. Both bindings must load one file.
RUN smi_library=$(python3 -c 'from importlib.metadata import distribution; print(distribution("rocm-sdk-core").locate_file("_rocm_sdk_core/lib/libamd_smi.so.27"))') && \
    cmp "${ROCM_PATH}/lib/libamd_smi.so" "${smi_library}" && \
    ln -sf "${smi_library}" "${ROCM_PATH}/lib/libamd_smi.so" && \
    ln -sf "${smi_library}" "${ROCM_PATH}/lib/libamd_smi.so.27"
