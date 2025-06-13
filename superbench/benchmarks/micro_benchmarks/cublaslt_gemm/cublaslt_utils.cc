// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>

#include <cuda_fp8.h>

#include "cublaslt_utils.h"

int GetScaleTensorSize(int inner, int outer, cublasLtMatmulMatrixScale_t scale_mode) {
    if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F) {
        return 1;
    }
    if (scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ||
        scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3) {
        const auto s_vscale = scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 32 : 16;
        const auto s_block_cols = 32;
        const auto s_block_rows = 4;
        const auto s_block_inner = 4;
        const auto block_rows = s_block_inner * s_vscale;
        const auto block_cols = s_block_cols * s_block_rows;
        const auto round_off = [](auto x, auto granularity) {
            return granularity * ((x + (granularity - 1)) / granularity);
        };
        const auto s_rows = round_off(inner, block_rows) / s_vscale;
        const auto s_cols = round_off(outer, block_cols);
        return s_rows * s_cols;
    }
    return 0;
}

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
        int64_t stridea = m * k, strideb = k * n, stridec = m * n, strided = m * n;
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

    // default to tf32 except for e5m2 inputs where the config is not supported
    cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    if (a_type == CUDA_R_8F_E5M2 || b_type == CUDA_R_8F_E5M2 || a_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3)
        gemm_compute_type = CUBLAS_COMPUTE_32F;
    if (a_type == CUDA_R_4F_E2M1 || b_type == CUDA_R_4F_E2M1)
        gemm_compute_type = CUBLAS_COMPUTE_32F;
    if (a_type == CUDA_R_64F || b_type == CUDA_R_64F)
        gemm_compute_type = CUBLAS_COMPUTE_64F;
    if (a_type == CUDA_R_8I)
        gemm_compute_type = CUBLAS_COMPUTE_32I;

    cublasLtMatmulDesc_t op_desc = nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, gemm_compute_type, CUDA_R_32F));
    op_desc_.reset(op_desc);

    if (a_type == CUDA_R_8F_E5M2 || b_type == CUDA_R_8F_E5M2 || a_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3) {
        // disable fastAccuMode, set to 0
        int8_t fastAccuMode = 1;
        cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(fastAccuMode));
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

    if (a_type == CUDA_R_4F_E2M1 || b_type == CUDA_R_4F_E2M1) {
        // Allocate and copy device scale values
        const auto a_scale = __nv_fp8_e4m3{1.f}, b_scale = __nv_fp8_e4m3{1.f}, d_out_scale = __nv_fp8_e4m3{1.f};
        const auto d_scale = 1.f;
        void *AscaleDev, *BscaleDev, *DscaleDev, *DOutscaleDev;

        // Set scale modes
        cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode,
                                                    sizeof(AScaleMode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode,
                                                    sizeof(BScaleMode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &DScaleMode,
                                                    sizeof(DScaleMode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,
                                                    &DOutScaleMode, sizeof(DOutScaleMode)));

        const auto a_scale_size =
            GetScaleTensorSize(transa != CUBLAS_OP_N ? k : m, transa != CUBLAS_OP_N ? m : k, AScaleMode);
        const auto b_scale_size =
            GetScaleTensorSize(transb != CUBLAS_OP_N ? n : k, transb != CUBLAS_OP_N ? k : n, BScaleMode);
        const auto d_scale_size = GetScaleTensorSize(m, n, DScaleMode);
        const auto d_out_scale_size = GetScaleTensorSize(m, n, DOutScaleMode);

        if (a_scale_size > 0) {
            __nv_fp8_e4m3 *a_scale_host = new __nv_fp8_e4m3[a_scale_size];
            std::fill_n(a_scale_host, a_scale_size, a_scale);
            cudaMalloc(&AscaleDev, a_scale_size * sizeof(__nv_fp8_e4m3));
            cudaMemcpy(AscaleDev, a_scale_host, a_scale_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
            delete[] a_scale_host;
        }
        if (b_scale_size > 0) {
            __nv_fp8_e4m3 *b_scale_host = new __nv_fp8_e4m3[b_scale_size];
            std::fill_n(b_scale_host, b_scale_size, b_scale);
            cudaMalloc(&BscaleDev, b_scale_size * sizeof(__nv_fp8_e4m3));
            cudaMemcpy(BscaleDev, b_scale_host, b_scale_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
            delete[] b_scale_host;
        }
        if (d_scale_size > 0) {
            float *d_scale_host = new float[d_scale_size];
            std::fill_n(d_scale_host, d_scale_size, d_scale);
            cudaMalloc(&DscaleDev, d_scale_size * sizeof(float));
            cudaMemcpy(DscaleDev, d_scale_host, d_scale_size * sizeof(float), cudaMemcpyHostToDevice);
            delete[] d_scale_host;
        }
        if (d_out_scale_size > 0) {
            __nv_fp8_e4m3 *d_out_scale_host = new __nv_fp8_e4m3[d_out_scale_size];
            std::fill_n(d_out_scale_host, d_out_scale_size, d_out_scale);
            cudaMalloc(&DOutscaleDev, d_out_scale_size * sizeof(__nv_fp8_e4m3));
            cudaMemcpy(DOutscaleDev, d_out_scale_host, d_out_scale_size * sizeof(__nv_fp8_e4m3),
                       cudaMemcpyHostToDevice);
            delete[] d_out_scale_host;
        }

        // Use device scale pointer attributes
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &AscaleDev,
                                                    sizeof(void *)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &BscaleDev,
                                                    sizeof(void *)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &DscaleDev,
                                                    sizeof(void *)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc_.get(), CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER,
                                                    &DOutscaleDev, sizeof(void *)));
    }
}

size_t cublasLtGemm::GetAlgorithm(int max_algorithm_count, size_t max_workspace_size) {
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference_.get(), CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &max_workspace_size, sizeof(max_workspace_size)));

    int found_algorithm_count = 0;
    std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithm_count);
    // Though we query all of possible algorithm, we will use the first later
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

void cublasLtGemm::Execute(void *matrix_a, void *matrix_b, void *matrix_c, void *matrix_d, float alpha, float beta,
                           void *workspace, size_t workspace_size, cudaStream_t stream) {
    CUBLAS_CHECK(cublasLtMatmul(handle_.get(), op_desc_.get(), static_cast<const void *>(&alpha), /* alpha */
                                matrix_a,                                                         /* A */
                                a_desc_.get(), matrix_b,                                          /* B */
                                b_desc_.get(), static_cast<const void *>(&beta),                  /* beta */
                                matrix_c,                                                         /* C */
                                c_desc_.get(), matrix_d,                                          /* D */
                                d_desc_.get(), &heuristic_results_.front().algo,                  /* algo */
                                workspace,                                                        /* workspace */
                                workspace_size, stream));                                         /* stream */
}
