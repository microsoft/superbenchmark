/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas-beta.h>
#include <rocblas/internal/rocblas_float8.h>
#include <math.h>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(status)                  \
    if(status != rocblas_status_success)              \
    {                                                 \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
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
            << "\t--datatype \t\tdatatype \tGEMM_STRIDED argument in out datatype:fp32,fp16,bf16\n"
            << "\t--num_warmups \t\t\tnum_warmups \t\tNumber of warmup runs\n"
            << "\t--num_iters \t\t\tnum_iters \t\tNumber of test runs\n"
            << std::endl;
}

static int ParseArguments(int argc, char *argv[], int64_t *m, int64_t *n, int64_t *k, float *alpha, float *beta,
                           int32_t *num_warmups, int32_t *num_iters) {
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
        } else if ((arg == "--num_warmups") && (i + 1 < argc)) {
          *num_warmups = atoi(argv[++i]);
        } else if ((arg == "--num_iters") && (i + 1 < argc)) {
          *num_iters = atoi(argv[++i]);
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



inline uint8_t random_fp8(){
    srand(1);

    return static_cast<uint8_t>((rand() % 7) - 3);
}

inline uint16_t random_fp16(){
    srand(1);

    return static_cast<uint16_t>((rand() % 7) - 3);
}

int main(int argc, char *argv[])
{
    // Init parameters with default values
    int64_t m = 80;
    int64_t n = 128;
    int64_t k = 128;
    float alpha = 1;
    float beta = 0;
    int32_t num_warmups = 10;
    int32_t num_iters = 50;

    if (ParseArguments(argc, argv, &m, &n, &k, &alpha, &beta, &num_warmups, &num_iters)) {
        ShowUsage(argv);
        return -1;
    }
    rocblas_datatype a_type       = rocblas_datatype_f8_r;
    rocblas_datatype b_type       = rocblas_datatype_f8_r;
    rocblas_datatype c_type       = rocblas_datatype_f16_r;
    rocblas_datatype d_type       = rocblas_datatype_f16_r;
    rocblas_computetype compute_type = rocblas_compute_type_f8_f8_f32;

    fprintf(stdout, "Parameters: m=%ld, n=%ld, k=%ld, alpha=%f, beta=%f, num_warmups=%d, num_iters=%d\n",
          m, n, k, alpha, beta, num_warmups, num_iters);
        
    using a_t = float;
    using b_t = float;
    using c_t = float;
    using d_t = float;

    

    int32_t           M = m, N = n, K = k;
    int32_t          A_row = m, A_col = k, B_row = k, B_col = n;


    size_t         lda, ldc = m, ldb=k, ldd = m;
    size_t         size_a1, size_b1, size_c1, size_d1;
    size_c1 = size_d1 = m * n;


    
    lda        = m;
    size_a1    = k * lda;
    ldb        = k;
    size_b1    = n * ldb;

    int batch_count =0;


    size_t size_a = batch_count == 0 ? size_a1 : size_a1 + 0 * (batch_count - 1);
    size_t size_b = batch_count == 0 ? size_b1 : size_b1 + 0 * (batch_count - 1);
    size_t size_c = batch_count == 0 ? size_c1 : size_c1 + 0 * (batch_count - 1);
    size_t size_d = batch_count == 0 ? size_d1 : size_d1 + 0 * (batch_count - 1);

    printf("size_a:%d, size_b:%d, size_c:%d, size_d:%d\n", size_a, size_b, size_c, size_d);

    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;

    printf("transa and transb done\n");
    std::vector<uint8_t> hA(size_a);
    std::vector<uint8_t> hB(size_b);
    std::vector<uint16_t>  hC(size_c);
    printf("init vector done\n");
    // Initial Data on CPU
    for (int i = 0; i < size_a; ++i) {
        hA[i] = random_fp8();
    }
    for (int i = 0; i < size_b; ++i) {
        hB[i] = random_fp8();
    }
    for (int i = 0; i < size_c; ++i) {
        hC[i] = random_fp16();
    
    }
    printf("init data done");

        // allocate memory on device
    void *dA, *dB, *dC, *dD;
     // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, size_a * sizeof(uint8_t)));
    CHECK_HIP_ERROR(hipMalloc(&dB, size_b * sizeof(uint8_t)));
    CHECK_HIP_ERROR(hipMalloc(&dC, size_c * sizeof(uint16_t)));
    CHECK_HIP_ERROR(hipMalloc(&dD, size_d * sizeof(uint16_t)));

    CHECK_HIP_ERROR(hipMemcpy(hA.data(), dA, size_a * sizeof(uint8_t), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(hB.data(), dB, size_b * sizeof(uint8_t), hipMemcpyHostToDevice));   
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, size_c * sizeof(uint16_t), hipMemcpyHostToDevice));
    printf("data copy done");

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    auto     algo           = rocblas_gemm_algo_solution_index;
    int32_t  solution_index = 0;
    uint32_t flags          = 0;
    rocblas_int* list_array = NULL;
    rocblas_int* list_size = NULL;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions_by_type( handle,
                                                                    a_type,
                                                                    c_type,
                                                                    c_type,
                                                                    flags,
                                                                    list_array,
                                                                    list_size))
    for (int i = 0; i < list_size[0]; i++)
    {
        solution_index = list_array[i];
        printf("solution %d is %d\n", i, list_array[i]);
    }
    printf("get solution done\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_warmups + num_iters; ++i) {
        if (i == num_warmups) {
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            start = std::chrono::high_resolution_clock::now();
        }
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex3(handle,
                                             transa,
                                             transb,
                                             m,
                                             n,
                                             k,
                                             &alpha,
                                             dA,
                                             a_type,
                                             lda,
                                             dB,
                                             b_type,
                                             ldb,                 
                                             &beta,
                                             dC,
                                             c_type,
                                             ldc,
                                             dD,
                                             d_type,
                                             ldd,
                                             compute_type,
                                             algo,
                                             solution_index,
                                             flags));
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // todo , cal gflops
    fprintf(stdout, "%lf\n", duration_us / num_iters);
    printf("%d\t%d\t%d\t%d\t%f\t%f\n", m, n, k, batch_count, duration_us,
           float(m) * float(n) * float(2 * k - 1) / 1e6 / duration_us * std::max(batch_count, 1));

    // copy output from device to CPU
    //CHECK_HIP_ERROR(hipMemcpy(&hD[0], dD, sizeof(uint16_t) * size_d, hipMemcpyDeviceToHost));


    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    CHECK_HIP_ERROR(hipFree(dD));
    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}