// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstdlib>
#include <ctime>
#include <getopt.h>
#include <memory>
#include <stdio.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cublaslt_utils.h"

using fp64 = double;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

struct Args {
    int m = 16;
    int n = 16;
    int k = 16;
    int batch = 0;
    int warmup = 20;
    int iter = 50;
    std::string in_type = "fp8e4m3";
};

void process_args(int argc, char **argv, Args *args) {
    const char *const short_opts = "m:n:k:b:w:i:t:";
    const option long_opts[] = {
        {"batch", required_argument, nullptr, 'b'},
        {"warmup", required_argument, nullptr, 'w'},
        {"iter", required_argument, nullptr, 'i'},
        {"in_type", required_argument, nullptr, 't'},
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'm':
            args->m = std::stoi(optarg);
            break;
        case 'n':
            args->n = std::stoi(optarg);
            break;
        case 'k':
            args->k = std::stoi(optarg);
            break;
        case 'b':
            args->batch = std::stoi(optarg);
            break;
        case 'w':
            args->warmup = std::stoi(optarg);
            break;
        case 'i':
            args->iter = std::stoi(optarg);
            break;
        case 't':
            args->in_type = std::string(optarg);
            break;
        }
    }
}

template <typename T> __global__ void init_matrix(T *matrix, const fp32 rand_val, const size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        // introduce more randomness
        fp32 val = rand_val;
        val = val + i * 3;

        matrix[i] = T(val);
    }
}

template <typename T> cudaDataType_t get_datatype() {
    if (std::is_same<T, fp64>::value)
        return CUDA_R_64F;
    if (std::is_same<T, fp32>::value)
        return CUDA_R_32F;
    if (std::is_same<T, fp16>::value)
        return CUDA_R_16F;
    if (std::is_same<T, bf16>::value)
        return CUDA_R_16BF;
    if (std::is_same<T, fp8e4m3>::value)
        return CUDA_R_8F_E4M3;
    if (std::is_same<T, fp8e5m2>::value)
        return CUDA_R_8F_E5M2;
    throw std::invalid_argument("Unknown type");
}

template <typename Ta, typename Tb, typename Tout>
float timing_matmul_tn(size_t m, size_t n, size_t k, size_t batch, int warmup, int iter) {
    // init matrix
    Ta *matrix_a = nullptr;
    Tb *matrix_b = nullptr;
    Tout *matrix_out = nullptr;
    batch = std::max<size_t>(batch, 1);
    cudaMalloc(&matrix_a, m * k * batch * sizeof(Ta));
    cudaMalloc(&matrix_b, k * n * batch * sizeof(Tb));
    cudaMalloc(&matrix_out, m * n * batch * sizeof(Tout));

    std::srand(std::time(nullptr));
    float rand_val = static_cast<float>(std::rand());

    init_matrix<Ta><<<216, 1024>>>(matrix_a, rand_val, m * k * batch);
 
    rand_val = static_cast<float>(std::rand());

    init_matrix<Tb><<<216, 1024>>>(matrix_b, rand_val, k * n * batch);

    // init gemm
    size_t lda = k, ldb = k, ldd = m;
    std::unique_ptr<cublasLtGemm> gemm = std::make_unique<cublasLtGemm>();
    gemm->Init();
    gemm->Setup(m, n, k, batch, lda, ldb, ldd, get_datatype<Ta>(), get_datatype<Tb>(), get_datatype<Tout>(),
                CUBLAS_OP_T, CUBLAS_OP_N, CUBLASLT_EPILOGUE_DEFAULT);

    void *workspace = nullptr;
    size_t workspace_size = gemm->GetAlgorithm(1, 2 * m * n);
    cudaMalloc(&workspace, workspace_size);

    // timer
    float time;
    cudaEvent_t startTime, endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    for (int i = 0; i < warmup; i++)
        gemm->Execute(reinterpret_cast<void *>(matrix_a), reinterpret_cast<void *>(matrix_b),
                      reinterpret_cast<void *>(matrix_out), reinterpret_cast<void *>(matrix_out), 1.f, 0.f, workspace,
                      workspace_size, 0);
    cudaEventRecord(startTime, 0);
    for (int i = 0; i < iter; i++)
        gemm->Execute(reinterpret_cast<void *>(matrix_a), reinterpret_cast<void *>(matrix_b),
                      reinterpret_cast<void *>(matrix_out), reinterpret_cast<void *>(matrix_out), 1.f, 0.f, workspace,
                      workspace_size, 0);
    cudaEventRecord(endTime, 0);
    cudaEventSynchronize(endTime);
    cudaEventElapsedTime(&time, startTime, endTime);

    // deallocate
    cudaFree(workspace);
    cudaFree(matrix_a);
    cudaFree(matrix_b);
    cudaFree(matrix_out);
    return (time * 1e3 / iter);
}

template <typename Ta, typename Tb = Ta, typename Tout = Ta> void run(Args *args) {
    float time_us = timing_matmul_tn<Ta, Tb, Tout>(args->m, args->n, args->k, args->batch, args->warmup, args->iter);
    // m n k batch time_us tflops
    printf("%d\t%d\t%d\t%d\t%f\t%f\n", args->m, args->n, args->k, args->batch, time_us,
           float(args->m) * float(args->n) * float(2 * args->k - 1) / 1e6 / time_us * std::max(args->batch, 1));
}

int main(int argc, char **argv) {
    Args args;
    process_args(argc, argv, &args);

    if (args.in_type == "fp64")
        run<fp64>(&args);
    else if (args.in_type == "fp32")
        run<fp32>(&args);
    else if (args.in_type == "fp16")
        run<fp16>(&args);
    else if (args.in_type == "bf16")
        run<bf16>(&args);
    else if (args.in_type == "fp8e4m3")
        run<fp8e4m3, fp8e4m3, fp16>(&args);
    else if (args.in_type == "fp8e5m2")
        run<fp8e5m2, fp8e4m3, fp16>(&args);
    else
        throw std::invalid_argument("Unknown type " + args.in_type);

    return 0;
}
