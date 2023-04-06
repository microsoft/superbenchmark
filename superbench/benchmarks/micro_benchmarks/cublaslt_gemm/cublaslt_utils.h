// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#include <cublasLt.h>

#define CUBLAS_CHECK(func)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = func;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            printf("cuBLAS call %s failed at %s:%d '%s'\n", #func, __FILE__, __LINE__, cublasGetStatusString(status)); \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

class cublasLtGemm {
  public:
    struct HandleDestroyer {
        void operator()(cublasLtHandle_t handle) const { cublasLtDestroy(handle); }
    };

    struct MatmulDescDestroyer {
        void operator()(cublasLtMatmulDesc_t matmul_desc) const { cublasLtMatmulDescDestroy(matmul_desc); }
    };

    struct LayoutDestroyer {
        void operator()(cublasLtMatrixLayout_t layout) const { cublasLtMatrixLayoutDestroy(layout); }
    };

    struct MatmulPreferenceDestroyer {
        void operator()(cublasLtMatmulPreference_t matmul_pref) const { cublasLtMatmulPreferenceDestroy(matmul_pref); }
    };

    using UniqueHandle = std::unique_ptr<std::remove_pointer<cublasLtHandle_t>::type, HandleDestroyer>;
    using UniqueOpDesc = std::unique_ptr<std::remove_pointer<cublasLtMatmulDesc_t>::type, MatmulDescDestroyer>;
    using UniqueLayoutDesc = std::unique_ptr<std::remove_pointer<cublasLtMatrixLayout_t>::type, LayoutDestroyer>;
    using UniqueMatmulPreference =
        std::unique_ptr<std::remove_pointer<cublasLtMatmulPreference_t>::type, MatmulPreferenceDestroyer>;

    void Init();

    void Setup(int m, int n, int k, int batch, int lda, int ldb, int ldd, cudaDataType_t a_type, cudaDataType_t b_type,
               cudaDataType_t d_type, cublasOperation_t transa, cublasOperation_t transb, cublasLtEpilogue_t epilogue,
               void *a_scale_inverse = nullptr, void *b_scale_inverse = nullptr);

    size_t GetAlgorithm(int max_algorithm_count, size_t max_workspace_size);

    void Execute(void *matrix_a, void *matrix_b, void *matrix_c, void *matrix_d, float alpha, float beta,
                 void *workspace, size_t workspace_size, cudaStream_t stream);

  private:
    UniqueHandle handle_;
    UniqueOpDesc op_desc_;
    UniqueLayoutDesc a_desc_;
    UniqueLayoutDesc b_desc_;
    UniqueLayoutDesc c_desc_;
    UniqueLayoutDesc d_desc_;
    UniqueMatmulPreference preference_;
    std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results_;
};
