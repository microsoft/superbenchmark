/*******************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__)
#include <hipblaslt/hipblaslt.h>
#include <rccl/rccl.h>
using cublasLtHalf = hipblasLtHalf;
#if defined(USE_HIPBLASLT_DATATYPE)
#define DIST_INF_HIP_DATATYPE_R_16F HIPBLASLT_R_16F
#define DIST_INF_HIP_DATATYPE_R_32F HIPBLASLT_R_32F
#else
#define DIST_INF_HIP_DATATYPE_R_16F HIPBLAS_R_16F
#define DIST_INF_HIP_DATATYPE_R_32F HIPBLAS_R_32F
#endif
#else
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <nccl.h>
using cublasLtHalf = half;
#endif

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(error)                                                                                        \
    if (error != cudaSuccess) {                                                                                        \
        fprintf(stderr, "Cuda error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error, __FILE__, __LINE__);      \
        exit(-1);                                                                                                      \
    }
#endif

#ifndef CHECK_CUBLASLT_ERROR
#define CHECK_CUBLASLT_ERROR(error)                                                                                    \
    if (error != CUBLAS_STATUS_SUCCESS) {                                                                              \
        fprintf(stderr, "cuBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__);                               \
        fprintf(stderr, "\n");                                                                                         \
        exit(-1);                                                                                                      \
    }
#endif

#ifndef CHECK_NCCL_ERROR
#define CHECK_NCCL_ERROR(error)                                                                                        \
    if (error != ncclSuccess) {                                                                                        \
        fprintf(stderr, "NCCL error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__);                                   \
        fprintf(stderr, "\n");                                                                                         \
        exit(-1);                                                                                                      \
    }
#endif

static void ShowUsage(char *argv[]) {
    std::cerr << "Usage: " << argv[0] << " <options>\n"
              << "options:\n"
              << "\t-h, --help\t\t\t\tShow this help message\n"
              << "\t-m \t\t\tm\t\tGEMM_STRIDED argument m\n"
              << "\t-n \t\t\tn\t\tGEMM_STRIDED argument n\n"
              << "\t-k \t\t\tk \t\tGEMM_STRIDED argument k\n"
              << "\t--alpha \t\talpha \t\tGEMM_STRIDED argument alpha\n"
              << "\t--beta \t\t\tbeta \t\tGEMM_STRIDED argument beta\n"
              << "\t--num_layers \t\t\tnum_layers \t\tNumber of layers in the model\n"
              << "\t--num_warmups \t\t\tnum_warmups \t\tNumber of warmup runs\n"
              << "\t--num_iters \t\t\tnum_iters \t\tNumber of test runs\n"
              << "\t--use_cuda_graph \t\t\tuse_cuda_graph \t\tWhether to launch kernels in CUDA graph mode\n"
              << std::endl;
}

static int ParseArguments(int argc, char *argv[], int64_t *m, int64_t *n, int64_t *k, float *alpha, float *beta,
                          int32_t *num_layers, int32_t *num_warmups, int32_t *num_iters, bool *use_cuda_graph) {
    if (argc >= 2) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if ((arg.at(0) == '-') || ((arg.at(0) == '-') && (arg.at(1) == '-'))) {
                if ((arg == "-h") || (arg == "--help")) {
                    return -1;
                } else if ((arg == "-m") && (i + 1 < argc)) {
                    *m = atoi(argv[++i]);
                } else if ((arg == "-n") && (i + 1 < argc)) {
                    *n = atoi(argv[++i]);
                } else if ((arg == "-k") && (i + 1 < argc)) {
                    *k = atoi(argv[++i]);
                } else if ((arg == "--alpha") && (i + 1 < argc)) {
                    *alpha = atof(argv[++i]);
                } else if ((arg == "--beta") && (i + 1 < argc)) {
                    *beta = atof(argv[++i]);
                } else if ((arg == "--num_layers") && (i + 1 < argc)) {
                    *num_layers = atoi(argv[++i]);
                } else if ((arg == "--num_warmups") && (i + 1 < argc)) {
                    *num_warmups = atoi(argv[++i]);
                } else if ((arg == "--num_iters") && (i + 1 < argc)) {
                    *num_iters = atoi(argv[++i]);
                } else if (arg == "--use_cuda_graph") {
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && (CUDART_VERSION >= 11030 || HIP_VERSION >= 50221310)
                    *use_cuda_graph = true;
#else
                    *use_cuda_graph = false;
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "not supported by current environment" << std::endl << std::endl;
                    return -1;
#endif
                } else {
                    std::cerr << "error with " << arg << std::endl;
                    std::cerr << "do not recognize option" << std::endl << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "error with " << arg << std::endl;
                std::cerr << "option must start with - or --" << std::endl << std::endl;
                return -1;
            }
        }
    }
    return 0;
}

void InitializeABCDEF(std::vector<cublasLtHalf> &ha, int64_t size_a, std::vector<cublasLtHalf> &hb, int64_t size_b,
                      std::vector<cublasLtHalf> &hc, int64_t size_c, std::vector<cublasLtHalf> &hd, int64_t size_d,
                      std::vector<cublasLtHalf> &he, int64_t size_e, std::vector<cublasLtHalf> &hf, int64_t size_f) {
    srand(1);
    for (int i = 0; i < size_a; ++i) {
        ha[i] = static_cast<cublasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < size_b; ++i) {
        hb[i] = static_cast<cublasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < size_c; ++i) {
        hc[i] = static_cast<cublasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < size_d; ++i) {
        hd[i] = static_cast<cublasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < size_e; ++i) {
        he[i] = static_cast<cublasLtHalf>((rand() % 7) - 3);
    }
    for (int i = 0; i < size_f; ++i) {
        hf[i] = static_cast<cublasLtHalf>((rand() % 7) - 3);
    }
}

// B[m, k] * A[k, n] + C[m, n] = D[m, n]
// E[k, m] * D[m, n] + F[k, n] = G[k, n]
void TestModel(int64_t m, int64_t n, int64_t k, float alpha, float beta, int32_t num_layers, int32_t num_warmups,
               int32_t num_iters, bool use_cuda_graph, ncclComm_t nccl_comm) {
    const int kNcclBufAlignment = 512;

    int size_a = k * n;
    int size_b = m * k;
    int size_c = m * n;
    int size_d = m * n;
    int size_e = k * m;
    int size_f = k * n;
    int size_g = (k * n + kNcclBufAlignment - 1) / kNcclBufAlignment * kNcclBufAlignment;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<cublasLtHalf> ha(size_a);
    std::vector<cublasLtHalf> hb(size_b);
    std::vector<cublasLtHalf> hc(size_c);
    std::vector<cublasLtHalf> hd(size_d);
    std::vector<cublasLtHalf> he(size_e);
    std::vector<cublasLtHalf> hf(size_f);
    std::vector<cublasLtHalf> hg(size_g);

    // initial data on host
    InitializeABCDEF(ha, size_a, hb, size_b, hc, size_c, hd, size_d, he, size_e, hf, size_f);

    // allocate memory on device
    void *da, *db, *dc, *dd, *de, *df, *dg;

    // Create stream
    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK_CUDA_ERROR(cudaMalloc(&da, size_a * sizeof(cublasLtHalf)));
    CHECK_CUDA_ERROR(cudaMalloc(&db, size_b * sizeof(cublasLtHalf)));
    CHECK_CUDA_ERROR(cudaMalloc(&dc, size_c * sizeof(cublasLtHalf)));
    CHECK_CUDA_ERROR(cudaMalloc(&dd, size_d * sizeof(cublasLtHalf)));
    CHECK_CUDA_ERROR(cudaMalloc(&de, size_e * sizeof(cublasLtHalf)));
    CHECK_CUDA_ERROR(cudaMalloc(&df, size_f * sizeof(cublasLtHalf)));
    CHECK_CUDA_ERROR(cudaMalloc(&dg, size_g * sizeof(cublasLtHalf)));
    // copy matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(da, ha.data(), sizeof(cublasLtHalf) * size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(db, hb.data(), sizeof(cublasLtHalf) * size_b, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dc, hc.data(), sizeof(cublasLtHalf) * size_c, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dd, hd.data(), sizeof(cublasLtHalf) * size_d, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(de, he.data(), sizeof(cublasLtHalf) * size_e, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(df, hf.data(), sizeof(cublasLtHalf) * size_f, cudaMemcpyHostToDevice));

    uint64_t workspace_size = 1024 * 1024;
    void *d_workspace;
    CHECK_CUDA_ERROR(cudaMalloc(&d_workspace, workspace_size));
    int returnedAlgoCount = 0;

    // cublasLt is not well supported by ROCm hipify tools, explicitly define ROCm logic instead.
#if defined(__HIP_PLATFORM_AMD__)
    hipblasLtHandle_t handle;
    hipblasLtMatrixLayout_t matA, matB, matC, matD, matE, matF, matG;
    hipblasLtMatmulDesc_t matmul1, matmul2;
    hipblasLtMatmulPreference_t pref;

    CHECK_CUBLASLT_ERROR(hipblasLtCreate(&handle));

    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, DIST_INF_HIP_DATATYPE_R_16F, k, n, k));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, DIST_INF_HIP_DATATYPE_R_16F, m, k, m));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, DIST_INF_HIP_DATATYPE_R_16F, m, n, m));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, DIST_INF_HIP_DATATYPE_R_16F, m, n, m));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matE, DIST_INF_HIP_DATATYPE_R_16F, k, m, k));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matF, DIST_INF_HIP_DATATYPE_R_16F, k, n, k));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matG, DIST_INF_HIP_DATATYPE_R_16F, k, n, k));

    CHECK_CUBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul1, HIPBLASLT_COMPUTE_F32, DIST_INF_HIP_DATATYPE_R_32F));
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul2, HIPBLASLT_COMPUTE_F32, DIST_INF_HIP_DATATYPE_R_32F));

    hipblasOperation_t trans = HIPBLAS_OP_N;
    CHECK_CUBLASLT_ERROR(
        hipblasLtMatmulDescSetAttribute(matmul1, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(int32_t)));
    CHECK_CUBLASLT_ERROR(
        hipblasLtMatmulDescSetAttribute(matmul1, HIPBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(int32_t)));
    CHECK_CUBLASLT_ERROR(
        hipblasLtMatmulDescSetAttribute(matmul2, HIPBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(int32_t)));
    CHECK_CUBLASLT_ERROR(
        hipblasLtMatmulDescSetAttribute(matmul2, HIPBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(int32_t)));

    // Set User Preference attributes
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                               &workspace_size, sizeof(workspace_size)));

    // Get Heuristic results
    hipblasLtMatmulHeuristicResult_t heuristicResult1[1] = {0};
    hipblasLtMatmulHeuristicResult_t heuristicResult2[1] = {0};
    // B[m, k] * A[k, n] + C[m, n] = D[m, n]
    // E[k, m] * D[m, n] + F[k, n] = G[k, n]
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul1, matB, matA, matC, matD, pref, 1,
                                                         heuristicResult1, &returnedAlgoCount));
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle, matmul2, matE, matD, matF, matG, pref, 1,
                                                         heuristicResult2, &returnedAlgoCount));
#else
    cublasLtHandle_t handle;
    cublasLtMatrixLayout_t matA, matB, matC, matD, matE, matF, matG;
    cublasLtMatmulDesc_t matmul1, matmul2;
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLASLT_ERROR(cublasLtCreate(&handle));

    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matA, CUDA_R_16F, k, n, k));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matB, CUDA_R_16F, m, k, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matC, CUDA_R_16F, m, n, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matD, CUDA_R_16F, m, n, m));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matE, CUDA_R_16F, k, m, k));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matF, CUDA_R_16F, k, n, k));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutCreate(&matG, CUDA_R_16F, k, n, k));

    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescCreate(&matmul1, CUBLAS_COMPUTE_16F, CUDA_R_32F));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescCreate(&matmul2, CUBLAS_COMPUTE_16F, CUDA_R_32F));

    cublasOperation_t trans = CUBLAS_OP_N;
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul1, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(int32_t)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul1, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(int32_t)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul2, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(int32_t)));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescSetAttribute(matmul2, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(int32_t)));

    // Set User Preference attributes
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &workspace_size, sizeof(workspace_size)));

    // Get Heuristic results
    cublasLtMatmulHeuristicResult_t heuristicResult1[1] = {0};
    cublasLtMatmulHeuristicResult_t heuristicResult2[1] = {0};
    // B[m, k] * A[k, n] + C[m, n] = D[m, n]
    // E[k, m] * D[m, n] + F[k, n] = G[k, n]
    CHECK_CUBLASLT_ERROR(cublasLtMatmulAlgoGetHeuristic(handle, matmul1, matB, matA, matC, matD, pref, 1,
                                                        heuristicResult1, &returnedAlgoCount));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulAlgoGetHeuristic(handle, matmul2, matE, matD, matF, matG, pref, 1,
                                                        heuristicResult2, &returnedAlgoCount));
#endif

    auto model_forward = [&] {
        for (int j = 0; j < num_layers; j++) {
            // B[m, k] * A[k, n] + C[m, n] = D[m, n]
            // E[k, m] * D[m, n] + F[k, n] = G[k, n]
            // cublasLt is not well supported by ROCm hipify tools, explicitly define ROCm logic instead.
#if defined(__HIP_PLATFORM_AMD__)
            CHECK_CUBLASLT_ERROR(hipblasLtMatmul(handle, matmul1, &alpha, db, matB, da, matA, &beta, dc, matC, dd, matD,
                                                 &heuristicResult1[0].algo, d_workspace, workspace_size, stream));
            CHECK_CUBLASLT_ERROR(hipblasLtMatmul(handle, matmul1, &alpha, de, matE, dd, matD, &beta, df, matF, dg, matG,
                                                 &heuristicResult2[0].algo, d_workspace, workspace_size, stream));
#else
            CHECK_CUBLASLT_ERROR(cublasLtMatmul(handle, matmul1, &alpha, db, matB, da, matA, &beta, dc, matC, dd, matD,
                                                &heuristicResult1[0].algo, d_workspace, workspace_size, stream));
            CHECK_CUBLASLT_ERROR(cublasLtMatmul(handle, matmul1, &alpha, de, matE, dd, matD, &beta, df, matF, dg, matG,
                                                &heuristicResult2[0].algo, d_workspace, workspace_size, stream));
#endif
            CHECK_NCCL_ERROR(ncclAllReduce(dg, dg, size_g, ncclFloat16, ncclSum, nccl_comm, stream));
        }
    };

#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && (CUDART_VERSION >= 11030 || HIP_VERSION >= 50221310)
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    if (use_cuda_graph) {
        CHECK_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        model_forward();
        CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA_ERROR(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    }
#endif

    std::chrono::steady_clock::time_point start_time, stop_time;
    std::vector<double> step_times(num_iters, 0.);
    for (int i = 0; i < num_warmups + num_iters; ++i) {
        if (i >= num_warmups) {
            start_time = std::chrono::steady_clock::now();
        }
#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && (CUDART_VERSION >= 11030 || HIP_VERSION >= 50221310)
        if (use_cuda_graph) {
            CHECK_CUDA_ERROR(cudaGraphLaunch(instance, stream));
        } else {
            model_forward();
        }
#else
        model_forward();
#endif
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (i >= num_warmups) {
            stop_time = std::chrono::steady_clock::now();
            double step_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time).count();
            step_times[i - num_warmups] = step_time;
        }
    }
    for (int i = 0; i < num_iters; i++) {
        fprintf(stdout, "Latency of step %d: %g ms\n", i, step_times[i] / 1e6);
    }

#if (NCCL_MAJOR > 2 || (NCCL_MAJOR >= 2 && NCCL_MINOR >= 9)) && (CUDART_VERSION >= 11030 || HIP_VERSION >= 50221310)
    // Destroy graph
    if (use_cuda_graph) {
        CHECK_CUDA_ERROR(cudaGraphExecDestroy(instance));
        CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
    }
#endif

    // Destroy stream
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    CHECK_CUDA_ERROR(cudaFree(da));
    CHECK_CUDA_ERROR(cudaFree(db));
    CHECK_CUDA_ERROR(cudaFree(dc));
    CHECK_CUDA_ERROR(cudaFree(dd));
    CHECK_CUDA_ERROR(cudaFree(de));
    CHECK_CUDA_ERROR(cudaFree(df));
    CHECK_CUDA_ERROR(cudaFree(dg));
    CHECK_CUDA_ERROR(cudaFree(d_workspace));
    // cublasLt is not well supported by ROCm hipify tools, explicitly define ROCm logic instead.
#if defined(__HIP_PLATFORM_AMD__)
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul1));
    CHECK_CUBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul2));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matE));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matF));
    CHECK_CUBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matG));
    CHECK_CUBLASLT_ERROR(hipblasLtDestroy(handle));
#else
    CHECK_CUBLASLT_ERROR(cublasLtMatmulPreferenceDestroy(pref));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescDestroy(matmul1));
    CHECK_CUBLASLT_ERROR(cublasLtMatmulDescDestroy(matmul2));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matA));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matB));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matC));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matD));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matE));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matF));
    CHECK_CUBLASLT_ERROR(cublasLtMatrixLayoutDestroy(matG));
    CHECK_CUBLASLT_ERROR(cublasLtDestroy(handle));
#endif

    return;
}

int main(int argc, char *argv[]) {
    // Init MPI
    int comm_rank, comm_size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Init NCCL
    int num_local_ranks = 0;
    ncclComm_t nccl_comm;
    ncclUniqueId nccl_id;
    if (comm_rank == 0) {
        CHECK_NCCL_ERROR(ncclGetUniqueId(&nccl_id));
    }
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&num_local_ranks))
    CHECK_CUDA_ERROR(cudaSetDevice(comm_rank % num_local_ranks));
    CHECK_NCCL_ERROR(ncclCommInitRank(&nccl_comm, comm_size, nccl_id, comm_rank));

    // Init parameters with default values
    int64_t m = 80;
    int64_t n = 128;
    int64_t k = 128;
    float alpha = 1;
    float beta = 1;
    int32_t num_layers = 50;
    int32_t num_warmups = 20;
    int32_t num_iters = 100;
    bool use_cuda_graph = false;

    if (ParseArguments(argc, argv, &m, &n, &k, &alpha, &beta, &num_layers, &num_warmups, &num_iters, &use_cuda_graph)) {
        ShowUsage(argv);
        return -1;
    }

    fprintf(stdout,
            "Parameters: m=%ld, n=%ld, k=%ld, alpha=%f, beta=%f, num_layers=%d, num_warmups=%d, num_iters=%d, "
            "use_cuda_graph=%d\n",
            m, n, k, alpha, beta, num_layers, num_warmups, num_iters, (int)use_cuda_graph);

    TestModel(m, n, k, alpha, beta, num_layers, num_warmups, num_iters, use_cuda_graph, nccl_comm);

    CHECK_NCCL_ERROR(ncclCommDestroy(nccl_comm));

    MPI_Finalize();

    return 0;
}
