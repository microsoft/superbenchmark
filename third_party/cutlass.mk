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
#   "f8"/"f4" element names mark runtime-typed kernels; the profiler takes the
#     actual type from --runtime_input_datatype_a/b.
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
# Optimized for peak-FLOPS benchmarking on GB200/GB300:
#   - Only large tiles (128x256, 256x128, 256x256) that are competitive at 16K³
#   - Only the canonical output type per precision (FP32-accum preferred)
#   - FP4 kept in full (~49 kernels, small enough)
#   - Block-scaled NVFP4 (true FP4 peak) added separately for SM103a (GB300)
#     via the BlockScaledGemm operation kind.  The FP4 Ultra path uses
#     cutlass3x_sm103_bstensorop_gemm_ue4m3x*_ultra_* kernels (ue4m3 scale,
#     vs16 = NVFP4 16-element block).  They have no f32 C operand, so the
#     f32-output variant is named f32_void_f32 (accumulator_C_D).
#   - Keeps all cluster/layout/schedule variants per tile so the profiler
#     can find the actual peak; runtime --kernels filtering in
#     cuda_gemm_flops_performance.py further narrows to ~36/precision.
# On aarch64, each sub-group must stay under ~1000 objects to avoid
# R_AARCH64_PREL32 relocation overflow in per-family shared libraries.
CUTLASS_KERNELS_SM100 := \
	cutlass3x_sm100_tensorop_tf32gemm_f32_f32_f32_f32_f32_*,\
	cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f16_f16_*,\
	cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_bf16_bf16_*,\
	cutlass3x_sm100_tensorop_gemm_s8_s8_s32_s8_s8_*,\
	cutlass3x_sm100_tensorop_gemm_e4m3_e4m3_f32_f32_f32_*,\
	cutlass3x_sm103_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_*ultra_2sm*,\
	cutlass3x_sm103_bstensorop_gemm_ue4m3xf4_ue4m3xf4_f32_void_f32_*ultra_2sm*

# SM107 Rubin 3x UMMA GEMM kernels (CUTLASS v4.8.0, "107a" build only)
#   - TF32/FP16/BF16: CUTLASS has no native SM107 generator; it retargets the
#     SM100 generators to Sm107 when a 107 arch is built, so the names match
#     the SM100 list with sm107.
#   - FP8 and NVFP4: native SM107 kernels (Rubin MMA shapes with B-matrix
#     reuse schedules).  Only native names contain "breuse" (or "no_breuse").
#     The FP8 ones are runtime-typed (f8_f8).  NVFP4 = e2m1 with ue4m3 scale
#     and 16-element blocks (vs16).
#   - No INT8: CuTe enables tcgen05 kind::i8 only for SM100a/SM101a/SM110.
CUTLASS_KERNELS_SM107 := \
	cutlass3x_sm107_tensorop_tf32gemm_f32_f32_f32_f32_f32_*,\
	cutlass3x_sm107_tensorop_gemm_f16_f16_f32_f16_f16_*,\
	cutlass3x_sm107_tensorop_gemm_bf16_bf16_f32_bf16_bf16_*,\
	cutlass3x_sm107_tensorop_gemm_f8_f8_f32_f32_f32_*breuse_*,\
	cutlass3x_sm107_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_*_o_vs16_*breuse_*

# Install prefix of the Rubin build; cuda_gemm_flops_performance.py uses it for CC 10.7.
CUTLASS_SM107_PREFIX = $(SB_MICRO_PATH)/cutlass_sm107

# Strip spaces that Make inserts from the line-continuation backslashes.
# Without this, cmake receives "cutlass_simt_dgemm..., cutlass_simt_sgemm..."
# (note the space after each comma) which breaks CUTLASS's kernel filter.
_strip_spaces = $(subst $(space),,$1)
space := $(subst ,, )

# $(call cutlass_build,<NVCC archs>,<kernel filter>,<install prefix>,<build dir>)
# Configures, builds and installs ./cutlass into <install prefix>/{bin,lib}.
cutlass_build = cmake -DCMAKE_INSTALL_BINDIR=$(3)/bin \
		-DCMAKE_INSTALL_LIBDIR=$(3)/lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCUTLASS_NVCC_ARCHS=$(1) \
		-DCUTLASS_ENABLE_EXAMPLES=OFF \
		-DCUTLASS_ENABLE_TESTS=OFF \
		-DCUTLASS_BUILD_SHARED_LIBS=ON \
		-DCUTLASS_BUILD_STATIC_LIBS=OFF \
		-DCMAKE_CXX_FLAGS="-fno-asynchronous-unwind-tables -fno-unwind-tables" \
		-DCMAKE_CUDA_FLAGS="-Xcompiler=-fno-asynchronous-unwind-tables -Xcompiler=-fno-unwind-tables" \
		-S ./cutlass \
		-B $(4) \
		"-DCUTLASS_LIBRARY_KERNELS=$(2)" && \
	cmake --build $(4) -j $(NUM_MAKE_JOBS) --target install && \
	rm -rf $(4)

# ---------------------------------------------------------------------------
# cuda_cutlass target
# ---------------------------------------------------------------------------
# Each CUDA version tier selects:
#   CUTLASS_TAG     — git tag to clone
#   CUTLASS_ARCHS   — semicolon-separated NVCC arch list
#   CUTLASS_KERNS   — comma-separated kernel filter patterns
#   CUTLASS_PATCHES — optional post-clone fixups (sed commands, etc.)
#   CUTLASS_SM107_KERNS — optional Rubin kernel filter; when set, a second
#                     "107a" build is installed into $(CUTLASS_SM107_PREFIX)
#
# After the version-specific block, a shared section clones, patches,
# configures, builds, and installs.

.PHONY: cuda_cutlass

cuda_cutlass:
ifeq ($(shell echo $(CUDA_VER)">=13.4" | bc -l), 1)
	# -----------------------------------------------------------------------
	# Blackwell GB200/GB300 + Rubin (SM107a) — CUTLASS v4.8.0, two folders
	# -----------------------------------------------------------------------
	# v4.8.0 is the first release with SM107.  With any 107 arch in the list,
	# it retargets the SM100 generators to SM107 (generator.py
	# resolve_sm10x_arch_range), so the SM100 kernels disappear from that build.
	# cutlass_profiler only runs kernels from its own libcutlass.so, so one
	# clone is built twice into separate folders:
	#   - "100a;103a" -> $(SB_MICRO_PATH)/{bin,lib}         (GB200, GB300)
	#   - "107a"      -> $(CUTLASS_SM107_PREFIX)/{bin,lib}  (Rubin)
	# v4.8.0 extended the INT8 gate to ['100f', '101f', '103a', '107f'].
	# Dropping '103a' keeps SM100a INT8 as in the 12.9 tier; the grep stops
	# the build if the patch no longer applies.
	$(eval CUTLASS_TAG     := v4.8.0)
	$(eval CUTLASS_ARCHS   := "100a;103a")
	$(eval CUTLASS_KERNS   := $(call _strip_spaces,$(CUTLASS_KERNELS_SM100)))
	$(eval CUTLASS_PATCHES := sed -i "s/arch_family_cc = \['100f', '101f', '103a', '107f'\]/arch_family_cc = ['100f', '101f', '107f']/" cutlass/python/cutlass_library/generator.py && grep -q "arch_family_cc = \['100f', '101f', '107f'\]" cutlass/python/cutlass_library/generator.py)
	$(eval CUTLASS_SM107_KERNS := $(call _strip_spaces,$(CUTLASS_KERNELS_SM107)))

else ifeq ($(shell echo $(CUDA_VER)">=12.9" | bc -l), 1)
	# -----------------------------------------------------------------------
	# Blackwell GB200 (SM100a) + GB300 (SM103a) — CUTLASS v4.3.5
	# -----------------------------------------------------------------------
	# Arch tokens must use the "a" suffix; bare "100"/"103" do not trigger
	# Blackwell code generation in CUTLASS v4.3.5's generator.py.
	#
	# SM103a (GB300 "Blackwell Ultra") does NOT support INT8 UMMA
	# (tcgen05.mma.kind::i8) — CuTe's config.hpp only defines
	# CUTE_ARCH_TCGEN05_S8_MMA_ENABLED for SM100a/SM101a.
	# However, CUTLASS's generator.py gates INT8 generation when ANY
	# of ['100f','101f','103a'] is in the arch list.  Since we build
	# with "100a;103a", the gate fires and skips INT8 for ALL archs
	# — including SM100a (GB200) which DOES support INT8.
	# Fix: remove only '103a' from the gate so INT8 is generated for
	# SM100a.  The SM103a cubin will compile INT8 with a runtime trap
	# (CUTE_INVALID_CONTROL_PATH) but the SM100a cubin will work.
	# On a GB300 the profiler will load the SM103a cubin and correctly
	# report the kernel as "failed"; cuda_gemm_flops_performance.py
	# excludes int8_tc from the SM10.3 kernel map so it is never run.
	$(eval CUTLASS_TAG     := v4.3.5)
	$(eval CUTLASS_ARCHS   := "100a;103a")
	$(eval CUTLASS_KERNS   := $(call _strip_spaces,$(CUTLASS_KERNELS_SM100)))
	$(eval CUTLASS_PATCHES := sed -i "s/arch_family_cc = \['100f', '101f', '103a'\]/arch_family_cc = ['100f', '101f']/" cutlass/python/cutlass_library/generator.py)

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
	# With the narrowed kernel filters (Blackwell ~3,800 objects, Rubin ~2,700;
	# max ~840 per sub-group), shared per-family libraries stay under the
	# R_AARCH64_PREL32 ±2 GB limit on aarch64.  Unwind table suppression
	# further reduces .eh_frame pressure.
	if [ -d cutlass ]; then rm -rf cutlass; fi
	git clone --branch $(CUTLASS_TAG) --depth 1 https://github.com/NVIDIA/cutlass.git
	$(CUTLASS_PATCHES)
	$(call cutlass_build,$(CUTLASS_ARCHS),$(CUTLASS_KERNS),$(SB_MICRO_PATH),./cutlass/build)
	$(if $(CUTLASS_SM107_KERNS),$(call cutlass_build,"107a",$(CUTLASS_SM107_KERNS),$(CUTLASS_SM107_PREFIX),./cutlass/build_sm107))
