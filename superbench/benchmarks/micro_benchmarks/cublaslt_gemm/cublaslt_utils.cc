// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cublaslt_utils.h"

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
        float a_scale = 1.0f, b_scale = 1.0f, d_scale = 1.0f, d_out_scale = 1.0f;
        void *AscaleDev, *BscaleDev, *DscaleDev, *DOutscaleDev;
        cudaMalloc(&AscaleDev, sizeof(float));
        cudaMalloc(&BscaleDev, sizeof(float));
        cudaMalloc(&DscaleDev, sizeof(float));
        cudaMalloc(&DOutscaleDev, sizeof(float));
        cudaMemcpy(AscaleDev, &a_scale, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(BscaleDev, &b_scale, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(DscaleDev, &d_scale, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(DOutscaleDev, &d_out_scale, sizeof(float), cudaMemcpyHostToDevice);

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
