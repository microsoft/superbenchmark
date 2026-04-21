# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License
#
# CUTLASS build configuration for SuperBench.
#
# This sub-makefile is included by the main Makefile and provides the
# cuda_cutlass target.  Moving CUTLASS here keeps the growing per-arch
# complexity (kernel filters, generator patches, naming conventions)
# isolated from the rest of the third-party build.
#
# ---------------------------------------------------------------------------
# Kernel naming conventions
# ---------------------------------------------------------------------------
#   SM70/80 legacy  : cutlass_simt_* / cutlass_tensorop_* (no arch prefix)
#   SM90/100+ (3x)  : cutlass3x_sm{arch}_tensorop_{core_name}_{dtype}_*
#     Note: TF32 kernels use "tf32gemm" (intermediate type before "gemm"),
#     e.g. cutlass3x_sm100_tensorop_tf32gemm_f32_f32_*
#
# ---------------------------------------------------------------------------
# Shared kernel lists (reusable fragments)
# ---------------------------------------------------------------------------
# SM70–90 legacy SIMT + TensorOp kernels
CUTLASS_KERNELS_LEGACY := \
	cutlass_simt_dgemm_128x128_8x2_*,\
	cutlass_simt_sgemm_128x128_8x2_*,\
	cutlass_simt_hgemm_256x128_8x2_*,\
	cutlass_tensorop_h884gemm_256x128_32x2_*,\
	cutlass_tensorop_d884gemm_128x128_16x3_*,\
	cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_*,\
	cutlass_tensorop_bf16_s16816gemm_bf16_256x128_32x3_*,\
	cutlass_tensorop_h16816gemm_256x128_32x3_*,\
	cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_*,\
	cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_*

# SM70–80 Volta/Turing subset (no Ampere+ TensorOp)
CUTLASS_KERNELS_VOLTA := \
	cutlass_simt_dgemm_128x128_8x2_*,\
	cutlass_simt_sgemm_128x128_8x2_*,\
	cutlass_simt_hgemm_256x128_8x2_*,\
	cutlass_tensorop_h884gemm_256x128_32x2_*

# SM100 Blackwell 3x UMMA GEMM kernels
CUTLASS_KERNELS_SM100 := \
	cutlass3x_sm100_tensorop_tf32gemm*,\
	cutlass3x_sm100_tensorop_gemm_f16_*,\
	cutlass3x_sm100_tensorop_gemm_bf16_*,\
	cutlass3x_sm100_tensorop_gemm_s8_*,\
	cutlass3x_sm100_tensorop_gemm_e4m3_*

# Strip spaces that Make inserts from the line-continuation backslashes.
# Without this, cmake receives "cutlass_simt_dgemm..., cutlass_simt_sgemm..."
# (note the space after each comma) which breaks CUTLASS's kernel filter.
_strip_spaces = $(subst $(space),,$1)
space := $(subst ,, )

# ---------------------------------------------------------------------------
# cuda_cutlass target
# ---------------------------------------------------------------------------
# Each CUDA version tier selects:
#   CUTLASS_TAG     — git tag to clone
#   CUTLASS_ARCHS   — semicolon-separated NVCC arch list
#   CUTLASS_KERNS   — comma-separated kernel filter patterns
#   CUTLASS_PATCHES — optional post-clone fixups (sed commands, etc.)
#
# After the version-specific block, a shared section clones, patches,
# configures, builds, and installs.

.PHONY: cuda_cutlass

cuda_cutlass:
ifeq ($(shell echo $(CUDA_VER)">=12.9" | bc -l), 1)
	# -----------------------------------------------------------------------
	# Blackwell GB200 (SM100a) + GB300 (SM103a) — CUTLASS v4.3.5
	# -----------------------------------------------------------------------
	# Arch tokens must use the "a" suffix; bare "100"/"103" do not trigger
	# Blackwell code generation in CUTLASS v4.3.5's generator.py.
	#
	# CUTLASS gates int8 GEMM generation when 103a is in the arch list
	# (arch_family_cc pruning heuristic). We patch that gate out because:
	#   1. SM103 hardware fully supports tcgen05.mma.kind::i8 (0.165 POPS dense).
	#   2. Generated int8 kernels use CC range [100, 110] (via ThorSMRenumbering),
	#      so CUTLASS runtime dispatches them on both SM100 and SM103.
	# The gate is a dead-kernel pruning optimization for family-binary builds,
	# not a hardware limitation. The sed patch only affects the two int8 gates
	# in GenerateSM100(); no other code paths use arch_family_cc.
	$(eval CUTLASS_TAG     := v4.3.5)
	$(eval CUTLASS_ARCHS   := "100a;103a")
	$(eval CUTLASS_KERNS   := $(call _strip_spaces,$(CUTLASS_KERNELS_SM100)))
	$(eval CUTLASS_PATCHES := sed -i "s/arch_family_cc = \['100f', '101f', '103a'\]/arch_family_cc = []/" cutlass/python/cutlass_library/generator.py)

else ifeq ($(shell echo $(CUDA_VER)">=12.8" | bc -l), 1)
	# -----------------------------------------------------------------------
	# Hopper SM90 + Blackwell SM100 — CUTLASS v3.9.2
	# -----------------------------------------------------------------------
	# SM90 non-FP8 precisions use legacy SM80 kernel names (SIMT + TensorOp)
	# which are backward-compatible on Hopper.  SM90 FP8 and SM100 use 3x naming.
	$(eval CUTLASS_TAG     := v3.9.2)
	$(eval CUTLASS_ARCHS   := "80;90;100")
	$(eval CUTLASS_KERNS   := $(call _strip_spaces,$(CUTLASS_KERNELS_LEGACY)),cutlass3x_sm90_tensorop_gemm_e4m3_*,$(call _strip_spaces,$(CUTLASS_KERNELS_SM100)))
	$(eval CUTLASS_PATCHES := @true)

else ifeq ($(shell echo $(CUDA_VER)">=11.8" | bc -l), 1)
	# -----------------------------------------------------------------------
	# Volta through Ada (SM70–SM90)
	# -----------------------------------------------------------------------
	$(eval CUTLASS_TAG     := v3.9.2)
	$(eval CUTLASS_ARCHS   := "70;75;80;86;89;90")
	$(eval CUTLASS_KERNS   := $(call _strip_spaces,$(CUTLASS_KERNELS_LEGACY)))
	$(eval CUTLASS_PATCHES := @true)

else
	# -----------------------------------------------------------------------
	# Volta/Turing subset (SM70–SM86)
	# -----------------------------------------------------------------------
	$(eval CUTLASS_TAG     := v3.9.2)
	$(eval CUTLASS_ARCHS   := "70;75;80;86")
	$(eval CUTLASS_KERNS   := $(call _strip_spaces,$(CUTLASS_KERNELS_VOLTA)))
	$(eval CUTLASS_PATCHES := @true)

endif
	# -----------------------------------------------------------------------
	# Shared clone → patch → build → install pipeline
	# -----------------------------------------------------------------------
	# On aarch64, per-family shared libraries (e.g. libcutlass_gemm_sm100_f16_gemm_e4m3.so)
	# can overflow R_AARCH64_PREL32 relocations in .eh_frame when thousands of FP8
	# kernel objects are linked into a single SO.  Work around this by:
	#   - Building static libs only (avoids the SO link that overflows)
	#   - Suppressing unwind tables in generated kernel TUs (reduces .eh_frame)
	# SuperBench only uses cutlass_profiler, which links statically, so no SO is needed.
	if [ -d cutlass ]; then rm -rf cutlass; fi
	git clone --branch $(CUTLASS_TAG) --depth 1 https://github.com/NVIDIA/cutlass.git
	$(CUTLASS_PATCHES)
	cmake -DCMAKE_INSTALL_BINDIR=$(SB_MICRO_PATH)/bin \
		-DCMAKE_INSTALL_LIBDIR=$(SB_MICRO_PATH)/lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCUTLASS_NVCC_ARCHS=$(CUTLASS_ARCHS) \
		-DCUTLASS_ENABLE_EXAMPLES=OFF \
		-DCUTLASS_ENABLE_TESTS=OFF \
		-DCUTLASS_BUILD_SHARED_LIBS=OFF \
		-DCUTLASS_BUILD_STATIC_LIBS=ON \
		-DCMAKE_CXX_FLAGS="-fno-asynchronous-unwind-tables" \
		-DCMAKE_CUDA_FLAGS="--compiler-options=-fno-asynchronous-unwind-tables" \
		-S ./cutlass \
		-B ./cutlass/build \
		"-DCUTLASS_LIBRARY_KERNELS=$(CUTLASS_KERNS)"
	cmake --build ./cutlass/build -j $(NUM_MAKE_JOBS) --target install
	rm -rf ./cutlass/build
