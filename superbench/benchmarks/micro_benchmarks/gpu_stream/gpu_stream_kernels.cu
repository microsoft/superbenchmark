// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file gpu_stream_kernels.cu
 * @brief CUDA kernel compilation unit for GPU stream benchmark.
 *
 * All template kernel implementations (CopyKernel, ScaleKernel, AddKernel, TriadKernel)
 * are defined in gpu_stream_kernels.hpp rather than here. This is required because:
 *
 * 1. **C++ Template Instantiation Model**: Templates are not compiled until they are
 *    instantiated with concrete types. The compiler needs to see the full template
 *    definition (not just declaration) at the point of instantiation.
 *
 * 2. **Separate Compilation Units**: When gpu_stream.cu calls `CopyKernel<double><<<...>>>`,
 *    nvcc needs the full kernel implementation visible in that translation unit.
 *    If implementations were only in this .cu file, gpu_stream.cu would only see
 *    declarations, causing "undefined reference" linker errors.
 *
 * 3. **CUDA-Specific Consideration**: Unlike regular C++ where explicit template
 *    instantiation in a .cpp file can work, CUDA kernel launches require the kernel
 *    code to be visible to nvcc when compiling the caller. This is because nvcc
 *    generates device code at compile time, not link time.
 *
 * 4. **Header Guards for Mixed Compilation**: The header uses `#ifdef __CUDACC__` to
 *    protect CUDA-specific code (blockIdx, threadIdx, __global__, etc.) from g++
 *    when the header is indirectly included by .cpp files (e.g., via gpu_stream.hpp).
 *
 * This file remains as the compilation unit that ensures the header is processed
 * by nvcc, and can host any future non-template helper functions if needed.
 */

#include "gpu_stream_kernels.hpp"