#!/usr/bin/env bash
# Build ROCm Transformer Engine and AITER without replacing the wheel SDK,
# PyTorch, or Triton. Invoke from a writable build directory inside the ROCm10
# image, not on the host. ROCM_PATH must select the wheel-SDK development root;
# CMake, Ninja, a C++ compiler, Git and Python development headers are required.
# Validated on Ubuntu 26.04 / Python 3.14.4 / MI300X (gfx942). The other two
# targets are compile-only coverage, not a runtime certification.
# For native-FP16 SB BERT/GPT2/Llama training (including FP8 linear layers),
# pass --optimizer_eps 1e-4: the legacy 1e-8 default underflows in FP16 Adam.
# BERT's pooler also requires an FP8-aligned batch size (use 16 or a multiple).
# GPT2's Hugging Face Conv1D layers are not converted to TE by SuperBench;
# use Llama/BERT or Megatron to exercise FP8 attention/MLP projection GEMMs.
# FP8 projection GEMMs do not imply FP8 dot-product attention support.
# For full-precision FP32 AITER Triton attention, set TRITON_F32_DEFAULT=ieee;
# its default TF32 dot products are independent of PyTorch's TF32 switches.
# Standalone aiter.flash_attn_func training requires return_lse=True; unpack
# (output, lse). Its CK JIT cold build can take many minutes, unlike TE's CK JIT.
# The separate Megatron checkout is selected with --code_base; the repository's
# older Megatron-LM and Megatron-DeepSpeed submodule pointers remain unchanged.
# An image can set SB_MEGATRON_PATH to that checkout as its non-DeepSpeed default.
set -euo pipefail

readonly TE_COMMIT=e7835ed1b134f56e58c575fba2455471064cab9b
readonly AITER_COMMIT=89a47b84ac4b576339a50047e290c383c2377389
readonly MEGATRON_COMMIT=4b4acac9a1d28ea6829c8d4f566d75698a21249d
# TE v2.15_rocm, AITER HEAD snapshot 2026-09-30, and Megatron Core v0.19.2.
# The tested TE 2.18-dev alternative cannot compile its BF16 small-sequence
# attention for gfx1250; do not silently replace this pin with moving dev.
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SOURCE_ROOT="${SB_ROCM10_FRAMEWORK_ROOT:-${PWD}/rocm10-frameworks}"
readonly WHEELHOUSE="${SOURCE_ROOT}/wheels/${TE_COMMIT}-${AITER_COMMIT}-${MEGATRON_COMMIT}"
mkdir -p "${SOURCE_ROOT}/scratch" "${WHEELHOUSE}"
export TMPDIR="${SOURCE_ROOT}/scratch"
export MAX_JOBS="${MAX_JOBS:-16}"
if ! [[ "${MAX_JOBS}" =~ ^[0-9]+$ ]] || ((MAX_JOBS < 1 || MAX_JOBS > 16)); then
    echo 'MAX_JOBS must be between 1 and 16.' >&2
    exit 1
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" NVTE_BUILD_MAX_JOBS="${MAX_JOBS}"
ROCM_PATH="$(rocm-sdk path --root)"
export ROCM_PATH ROCM_HOME="${ROCM_PATH}" HIP_PATH="${ROCM_PATH}" RCCL_HOME="${ROCM_PATH}"
export NVTE_USE_ROCM=1 NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH='gfx942;gfx950;gfx1250'
export NVTE_NO_LOCAL_VERSION=1
export NVTE_CK_JIT=1 NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=1
export AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 BUILD_TARGET=rocm

# An explicit constraints file also protects transitive dependency resolution.
# --no-build-isolation prevents pip from fetching a different build-time torch.
python - "${SOURCE_ROOT}/sdk-constraints.txt" <<'PY'
import importlib.metadata as metadata
import sys

expected = {
    "torch": "2.13.0+rocm10.0.0",
    "triton": "3.8.0+git4cff872c.rocm10.0.0",
    "rocm-sdk-core": "10.0.0",
    "rocm-sdk-devel": "10.0.0",
}
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise SystemExit(f"Expected {name}=={version}, found {actual}")
protected = []
for distribution in metadata.distributions():
    name = distribution.metadata["Name"].lower().replace("_", "-")
    if name.startswith("rocm-") or name in {"rocm", "torch", "torchvision", "torchaudio", "triton", "apex"}:
        protected.append(f"{name}=={distribution.version}")
with open(sys.argv[1], "w") as file:
    file.write("\n".join(sorted(protected)) + "\n")
PY
export PIP_CONSTRAINT="${SOURCE_ROOT}/sdk-constraints.txt"

checkout() {
    local repository="$1" commit="$2" destination="$3"
    if [[ ! -d "${destination}/.git" ]]; then
        git init --quiet "${destination}"
        git -C "${destination}" remote add origin "${repository}"
    fi
    git -C "${destination}" fetch --quiet --depth 1 origin "${commit}"
    git -C "${destination}" checkout --quiet --detach "${commit}"
    git -C "${destination}" submodule update --init --recursive --depth 1
}

python -m pip install \
    pybind11==3.0.1 pydantic==2.13.5 einops==0.8.2 psutil==7.2.2 mpi4py==4.1.1 \
    flydsl==0.3.4.1 pandas==3.0.5 nvdlfw-inspect==0.2.2 \
    onnxscript==0.7.2 onnx-ir==1.0.0 onnx==1.23.1 ml_dtypes==0.6.0
checkout https://github.com/ROCm/TransformerEngine.git "${TE_COMMIT}" "${SOURCE_ROOT}/TransformerEngine"
checkout https://github.com/ROCm/aiter.git "${AITER_COMMIT}" "${SOURCE_ROOT}/aiter"
checkout https://github.com/NVIDIA/Megatron-LM.git "${MEGATRON_COMMIT}" "${SOURCE_ROOT}/Megatron-LM"
MEGATRON_PATCH="${SB_ROCM10_MEGATRON_PATCH:-${SCRIPT_DIR}/../../third_party/Megatron/megatron_core_0.19.2_rocm10.patch}"
if ! git -C "${SOURCE_ROOT}/Megatron-LM" apply --reverse --check "${MEGATRON_PATCH}" 2>/dev/null; then
    git -C "${SOURCE_ROOT}/Megatron-LM" apply --check "${MEGATRON_PATCH}"
    git -C "${SOURCE_ROOT}/Megatron-LM" apply "${MEGATRON_PATCH}"
fi

# TE pins its own private AITER/CK implementation in qola_manifest.toml.
# Do not override it with the separately installed Python AITER package.
# CK excludes gfx1250 upstream; common TE kernels still compile for all three
# requested targets. AOTriton ships images for gfx942/gfx950, not gfx1250.
python -m pip wheel --no-build-isolation --no-deps \
    "${SOURCE_ROOT}/TransformerEngine" "${SOURCE_ROOT}/aiter" "${SOURCE_ROOT}/Megatron-LM" \
    -w "${WHEELHOUSE}"
python -m pip install --no-deps "${WHEELHOUSE}"/*.whl

python -m pip check
python - "${SOURCE_ROOT}/sdk-constraints.txt" <<'PY'
import importlib.metadata as metadata
import sys

for line in open(sys.argv[1]):
    name, expected = line.strip().split("==")
    actual = metadata.version(name)
    if actual != expected:
        raise SystemExit(f"Protected dependency changed: {name}: {expected} -> {actual}")
PY
python - "${SOURCE_ROOT}" "${TE_COMMIT}" "${AITER_COMMIT}" "${MEGATRON_COMMIT}" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

root, te, aiter, megatron = sys.argv[1:]
manifest = {
    "sources": {"transformer_engine": te, "aiter": aiter, "megatron": megatron},
    "packages": {name: metadata.version(name) for name in
                 ["torch", "triton", "transformer_engine", "amd-aiter", "megatron-core", "flydsl"]},
    "compiled_te_targets": ["gfx942", "gfx950", "gfx1250"],
    "ck_attention_targets": ["gfx942", "gfx950"],
    "aotriton_image_targets": ["gfx942", "gfx950"],
}
(Path(root) / "framework-pins.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
echo "Framework sources: ${SOURCE_ROOT}"
echo "Megatron --code_base: ${SOURCE_ROOT}/Megatron-LM"
echo "Optional image ENV: SB_MEGATRON_PATH=${SOURCE_ROOT}/Megatron-LM"
