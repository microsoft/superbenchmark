// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cublaslt_utils.h"
#include <algorithm> // for std::sort
#include <cassert>   // for assert
#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <limits>
#include <string>

#if CUDA_VERSION >= 12080
#ifndef SUPERBENCH_HAS_MXFP8_MN_K4_SCALE
#define SUPERBENCH_HAS_MXFP8_MN_K4_SCALE 0
#endif

constexpr size_t DivideRoundUp(size_t value, size_t divisor) {
    return value / divisor + static_cast<size_t>(value % divisor != 0);
}

constexpr size_t RoundUp(size_t value, size_t alignment) { return DivideRoundUp(value, alignment) * alignment; }

constexpr size_t GetMnK4ScaleTensorSize(size_t inner, size_t outer, size_t vector_size) {
    return RoundUp(outer, 4) * DivideRoundUp(inner, vector_size);
}

static_assert(GetMnK4ScaleTensorSize(160, 5, 32) == 40, "Unexpected VEC32 MN_K4 scale tensor size");
static_assert(GetMnK4ScaleTensorSize(257, 5, 128) == 24, "Unexpected VEC128 MN_K4 scale tensor size");

size_t GetScaleTensorSize(size_t inner, size_t outer, cublasLtMatmulMatrixScale_t scale_mode) {
    if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) {
        return 1;
    }
#if SUPERBENCH_HAS_MXFP8_MN_K4_SCALE
    if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_MN_K4_UE8M0) {
        return GetMnK4ScaleTensorSize(inner, outer, 32);
    }
    if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_MN_K4_UE8M0) {
        return GetMnK4ScaleTensorSize(inner, outer, 128);
    }
#endif
    if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 ||
        scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) {
        size_t s_vscale = 16;
        if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) {
            s_vscale = 32;
        }
        constexpr size_t s_block_cols = 32;
        constexpr size_t s_block_rows = 4;
        constexpr size_t s_block_inner = 4;
        const auto block_rows = s_block_inner * s_vscale;
        const auto block_cols = s_block_cols * s_block_rows;
        const auto s_rows = RoundUp(inner, block_rows) / s_vscale;
        const auto s_cols = RoundUp(outer, block_cols);
        return s_rows * s_cols;
    }
    return 0;
}
#endif

cublasLtGemm::~cublasLtGemm() {
#if CUDA_VERSION >= 12080
    ClearScaleBuffers();
#endif
}

#if CUDA_VERSION >= 12080
void cublasLtGemm::ClearScaleBuffers() {
    for (auto *buffer : scale_buffers_) {
        cudaFree(buffer);
    }
    scale_buffers_.clear();
}

void cublasLtGemm::AllocateScaleBuffer(size_t element_count, size_t element_size, const void *fill_value,
                                       void **device_pointer) {
    if (element_count == 0 || element_size == 0) {
        throw std::runtime_error("Invalid cuBLASLt scale tensor size");
    }
    if (element_count > std::numeric_limits<size_t>::max() / element_size) {
        throw std::overflow_error("cuBLASLt scale tensor size exceeds addressable memory");
    }

    std::vector<uint8_t> host_buffer(element_count * element_size);
    for (size_t i = 0; i < element_count; ++i) {
        std::copy_n(static_cast<const uint8_t *>(fill_value), element_size, host_buffer.data() + i * element_size);
    }

    auto status = cudaMalloc(device_pointer, host_buffer.size());
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc for cuBLASLt scale tensor failed: " +
                                 std::string(cudaGetErrorString(status)));
    }

    status = cudaMemcpy(*device_pointer, host_buffer.data(), host_buffer.size(), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(*device_pointer);
        *device_pointer = nullptr;
        throw std::runtime_error("cudaMemcpy for cuBLASLt scale tensor failed: " +
                                 std::string(cudaGetErrorString(status)));
    }

    try {
        scale_buffers_.push_back(*device_pointer);
    } catch (...) {
        cudaFree(*device_pointer);
        *device_pointer = nullptr;
        throw;
    }
}

void cublasLtGemm::SetupScaleModes(const MatrixScaleModes &scale_modes, cublasOperation_t transa,
                                   cublasOperation_t transb) {
    if (!scale_modes.enabled) {
        return;
    }

    ClearScaleBuffers();

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_modes.a,
                                                sizeof(scale_modes.a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_modes.b,
                                                sizeof(scale_modes.b)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scale_modes.d,
                                                sizeof(scale_modes.d)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,
                                                &scale_modes.d_out, sizeof(scale_modes.d_out)));

    const auto a_scale_size =
        GetScaleTensorSize(transa != CUBLAS_OP_N ? k_ : m_, transa != CUBLAS_OP_N ? m_ : k_, scale_modes.a);
    const auto b_scale_size =
        GetScaleTensorSize(transb != CUBLAS_OP_N ? n_ : k_, transb != CUBLAS_OP_N ? k_ : n_, scale_modes.b);
    const auto d_scale_size = GetScaleTensorSize(m_, n_, scale_modes.d);
    const auto d_out_scale_size = GetScaleTensorSize(m_, n_, scale_modes.d_out);

    void *a_scale_dev = nullptr, *b_scale_dev = nullptr, *d_scale_dev = nullptr, *d_out_scale_dev = nullptr;
    const auto ue4m3_one = __nv_fp8_e4m3{1.f};
    const auto ue8m0_one = uint8_t{127};
    const auto fp32_one = 1.f;

    if (scale_modes.a == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
        AllocateScaleBuffer(a_scale_size, sizeof(ue4m3_one), &ue4m3_one, &a_scale_dev);
    } else {
        AllocateScaleBuffer(a_scale_size, sizeof(ue8m0_one), &ue8m0_one, &a_scale_dev);
    }
    if (scale_modes.b == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
        AllocateScaleBuffer(b_scale_size, sizeof(ue4m3_one), &ue4m3_one, &b_scale_dev);
    } else {
        AllocateScaleBuffer(b_scale_size, sizeof(ue8m0_one), &ue8m0_one, &b_scale_dev);
    }
    AllocateScaleBuffer(d_scale_size, sizeof(fp32_one), &fp32_one, &d_scale_dev);
    if (scale_modes.d_out == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
        AllocateScaleBuffer(d_out_scale_size, sizeof(ue4m3_one), &ue4m3_one, &d_out_scale_dev);
    } else {
        AllocateScaleBuffer(d_out_scale_size, sizeof(ue8m0_one), &ue8m0_one, &d_out_scale_dev);
    }

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_dev,
                                                sizeof(void *)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_dev,
                                                sizeof(void *)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_dev,
                                                sizeof(void *)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER,
                                                &d_out_scale_dev, sizeof(void *)));
}
#endif

void cublasLtGemm::Init() {
    cublasLtHandle_t handle;
    CUBLAS_CHECK(cublasLtCreate(&handle));
    handle_.reset(handle);

    /* preference can be initialized without arguments */
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    preference_.reset(preference);
}

void cublasLtGemm::Setup(int m, int n, int k, int batch, int lda, int ldb, int ldc, int ldd, cudaDataType_t a_type,
                         cudaDataType_t b_type, cudaDataType_t c_type, cudaDataType_t d_type, cublasOperation_t transa,
                         cublasOperation_t transb, cublasLtEpilogue_t epilogue,
                         void *a_scale_inverse, /* only need to be set for fp8 */
                         void *b_scale_inverse  /* only need to be set for fp8 */
) {
    // Store dimensions
    m_ = m;
    n_ = n;
    k_ = k;

    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr, d_desc = nullptr;
    // Create matrix descriptors.
    CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&a_desc, a_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CUBLAS_CHECK(
        cublasLtMatrixLayoutCreate(&b_desc, b_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, c_type, m, n, ldc));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d_desc, d_type, m, n, ldd));

    // strided batch gemm
    if (batch > 0) {
        int64_t stridea = static_cast<int64_t>(m) * k, strideb = static_cast<int64_t>(k) * n,
                stridec = static_cast<int64_t>(m) * n, strided = static_cast<int64_t>(m) * n;
        CUBLAS_CHECK(
            cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
                                                      sizeof(stridea)));
        CUBLAS_CHECK(
            cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
                                                      sizeof(strideb)));
        CUBLAS_CHECK(
            cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
                                                      sizeof(stridec)));
        CUBLAS_CHECK(
            cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strided,
                                                      sizeof(strided)));
    }
    a_desc_.reset(a_desc);
    b_desc_.reset(b_desc);
    c_desc_.reset(c_desc);
    d_desc_.reset(d_desc);

    // Set compute type and scale type based on input types
    cublasComputeType_t gemm_compute_type;
    cudaDataType_t scale_type;
    if (a_type == CUDA_R_8F_E5M2 || b_type == CUDA_R_8F_E5M2 || a_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3) {
        gemm_compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
    } else if (a_type == CUDA_R_16F || b_type == CUDA_R_16F || a_type == CUDA_R_16BF || b_type == CUDA_R_16BF) {
        gemm_compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
#if CUDA_VERSION >= 12080
    } else if (a_type == CUDA_R_4F_E2M1 || b_type == CUDA_R_4F_E2M1) {
        gemm_compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
#endif
#if CUDA_VERSION >= 12080 && __has_include(<cuda_fp6.h>)
    } else if (a_type == CUDA_R_6F_E2M3 || b_type == CUDA_R_6F_E2M3 || a_type == CUDA_R_6F_E3M2 ||
               b_type == CUDA_R_6F_E3M2) {
        gemm_compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
#endif
    } else if (a_type == CUDA_R_64F || b_type == CUDA_R_64F) {
        gemm_compute_type = CUBLAS_COMPUTE_64F;
        scale_type = CUDA_R_64F;
    } else if (a_type == CUDA_R_8I) {
        gemm_compute_type = CUBLAS_COMPUTE_32I;
        scale_type = CUDA_R_32I;
    } else {
        gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        scale_type = CUDA_R_32F;
    }

    cublasLtMatmulDesc_t op_desc = nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, gemm_compute_type, scale_type));
    op_desc_.reset(op_desc);

    if (a_type == CUDA_R_8F_E5M2 || b_type == CUDA_R_8F_E5M2 || a_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3) {
        int8_t fastAccuMode = 1;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode,
                                                    sizeof(fastAccuMode)));
    }

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    if (a_scale_inverse != nullptr) {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &a_scale_inverse, sizeof(a_scale_inverse)));
    }
    if (b_scale_inverse != nullptr) {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &b_scale_inverse, sizeof(b_scale_inverse)));
    }
    CUBLAS_CHECK(
        cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
}

size_t cublasLtGemm::GetAlgorithm(int max_algorithm_count, size_t max_workspace_size) {
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference_.get(), CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &max_workspace_size, sizeof(max_workspace_size)));
    int found_algorithm_count = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithm_count);
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_.get(), op_desc_.get(), a_desc_.get(), b_desc_.get(),
                                                c_desc_.get(), d_desc_.get(), preference_.get(), max_algorithm_count,
                                                results.data(), &found_algorithm_count));
    if (found_algorithm_count == 0) {
        throw std::runtime_error("Unable to find any suitable algorithms");
    }

    results.resize(found_algorithm_count);
    heuristic_results_ = std::move(results);
    return heuristic_results_.front().workspaceSize;
}

size_t cublasLtGemm::GetAlgorithmExhaustive(int max_algorithm_count, size_t max_workspace_size, float alpha, float beta,
                                            void *matrix_a, void *matrix_b, void *matrix_c, void *matrix_d,
                                            int repeat_iterations, int warmup_iterations) {
    // Set workspace size in preference
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference_.get(), CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &max_workspace_size, sizeof(max_workspace_size)));

    // Get heuristic algorithms
    int found_algorithm_count = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithm_count);
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(handle_.get(), op_desc_.get(), a_desc_.get(), b_desc_.get(),
                                                c_desc_.get(), d_desc_.get(), preference_.get(), max_algorithm_count,
                                                results.data(), &found_algorithm_count));
    if (found_algorithm_count == 0) {
        throw std::runtime_error("Unable to find any suitable algorithms");
    }

    results.resize(found_algorithm_count);
    heuristic_results_ = std::move(results);

    // Create stream and events for timing
    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    cudaStreamCreate(&stream);
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Test each algorithm multiple times to find the best one
    std::vector<float> algoTimes(repeat_iterations);

    // Allocate workspace
    void *workspace = nullptr;
    cudaMalloc(&workspace, max_workspace_size);

    // Test each algorithm
    algo_metrics_.clear();
    algo_metrics_.reserve(found_algorithm_count);

    for (int algoIdx = 0; algoIdx < found_algorithm_count; algoIdx++) {
        // Skip algorithms that require more workspace than available
        if (heuristic_results_[algoIdx].workspaceSize > max_workspace_size) {
            continue;
        }

        // warmup
        for (int warmupIdx = 0; warmupIdx < warmup_iterations; warmupIdx++) {
            cublasStatus_t status =
                cublasLtMatmul(handle_.get(), op_desc_.get(), &alpha, matrix_a, a_desc_.get(), matrix_b, b_desc_.get(),
                               &beta, matrix_c, c_desc_.get(), matrix_d, d_desc_.get(),
                               &heuristic_results_[algoIdx].algo, workspace, max_workspace_size, stream);
        }

        // Test each algorithm multiple times
        cudaEventRecord(startEvent, stream);
        for (int checkIdx = 0; checkIdx < repeat_iterations; checkIdx++) {
            cublasStatus_t status =
                cublasLtMatmul(handle_.get(), op_desc_.get(), &alpha, matrix_a, a_desc_.get(), matrix_b, b_desc_.get(),
                               &beta, matrix_c, c_desc_.get(), matrix_d, d_desc_.get(),
                               &heuristic_results_[algoIdx].algo, workspace, max_workspace_size, stream);

            // Skip if algorithm fails
            if (status != CUBLAS_STATUS_SUCCESS) {
                algoTimes[checkIdx] = std::numeric_limits<float>::max();
                continue;
            }
        }

        cudaEventRecord(stopEvent, stream);
        cudaEventSynchronize(stopEvent);

        float time = 0;
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        algoTimes[algoIdx] = time / repeat_iterations;

        float meanTime = algoTimes[algoIdx];
        float flops = 2.0f * m_ * n_ * k_ / (meanTime * 1e-3f);

        // Store metrics
        AlgorithmMetrics metrics;
        metrics.algo = heuristic_results_[algoIdx].algo;
        metrics.workspace_size = heuristic_results_[algoIdx].workspaceSize;
        metrics.time = meanTime;
        metrics.flops = flops;
        algo_metrics_.push_back(metrics);
    }

    std::sort(algo_metrics_.begin(), algo_metrics_.end(),
              [](const AlgorithmMetrics &a, const AlgorithmMetrics &b) { return a.time < b.time; });

    if (!algo_metrics_.empty())
        heuristic_results_[0].algo = algo_metrics_.front().algo;

    // Clean up resources
    cudaFree(workspace);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaStreamDestroy(stream);

    if (!algo_metrics_.empty()) {
        return algo_metrics_.front().workspace_size;
    }

    throw std::runtime_error("No valid algorithms found during autotune");
}

void cublasLtGemm::Execute(void *matrix_a, void *matrix_b, void *matrix_c, void *matrix_d, float alpha, float beta,
                           void *workspace, size_t workspace_size, cudaStream_t stream) {

    CUBLAS_CHECK(cublasLtMatmul(handle_.get(), op_desc_.get(), static_cast<const void *>(&alpha), /* alpha */
                                matrix_a,                                                         /* A */
                                a_desc_.get(), matrix_b,                                          /* B */
                                b_desc_.get(), static_cast<const void *>(&beta),                  /* beta */
                                matrix_c,                                                         /* C */
                                c_desc_.get(), matrix_d,                                          /* D */
                                d_desc_.get(), &heuristic_results_.front().algo, workspace,       /* workspace */
                                workspace_size, stream));                                         /* stream */
}
