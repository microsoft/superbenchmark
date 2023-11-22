// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cublas_benchmark.h
 * @brief Unify a base class for cublas function benchmark
 */

#pragma once

#include <chrono>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "cublas_helper.h"

/**
 * @brief Enum of cublas function name
 */
enum cublas_function_name_enum {
    e_cublasSgemm = 0,
    e_cublasCgemm,
    e_cublasGemmEx,
    e_cublasGemmStridedBatchedEx,
    e_cublasSgemmStridedBatched,
    e_cublasCgemm3mStridedBatched
};

/**
 * @brief Map from cublas function name to cublas function name enum
 */
static std::unordered_map<std::string, cublas_function_name_enum> const cublas_function_name_string = {
    {"cublasSgemm", cublas_function_name_enum::e_cublasSgemm},
    {"cublasCgemm", cublas_function_name_enum::e_cublasCgemm},
    {"cublasGemmEx", cublas_function_name_enum::e_cublasGemmEx},
    {"cublasGemmStridedBatchedEx", cublas_function_name_enum::e_cublasGemmStridedBatchedEx},
    {"cublasSgemmStridedBatched", cublas_function_name_enum::e_cublasSgemmStridedBatched},
    {"cublasCgemm3mStridedBatched", cublas_function_name_enum::e_cublasCgemm3mStridedBatched},
};

/**
 * @brief Class to store params of cublas function and run the benchmark of this function
 */
class CublasFunction {
  protected:
    int num_test;                      ///< the number of steps used to test and measure
    int warm_up;                       ///< the number of steps used to warm up
    int num_in_step;                   ///< the number of functions invoking in a step
    int random_seed;                   ///< the random seed used to generate random data
    double eps;                        ///< the acceptable error bound for numeric stability
    bool correctness;                  ///< whether enable correctness check or not
    bool random_data;                  ///< whether enable random data generation or not
    std::string name_;                 ///< the name of the cublas function
    int m_;                            ///< the m dim of matrix
    int k_;                            ///< the k dim of matrix
    int n_;                            ///< the n dim of matrix
    int transa_;                       ///< whether the first matrix transpose
    int transb_;                       ///< whether the second matrix transpose
    std::string datatype_;             ///< data type used in cublasGemmEx and cublasGemmStridedBatchedEx
    bool use_tensor_core_;             ///< choose the algo used in cublasGemmEx and cublasGemmStridedBatchedEx
    int batch_count_;                  ///< the number of the batch used in some cublas function
    cublas_function_name_enum e_name_; ///< enum cublas functin name
    std::string function_str_;         ///< the str representing the cublas function with params
    cublasHandle_t cublas_handle;      ///< the handle of cublas function

    /**
     * @brief Fill the random data into the input
     */
    template <typename T> void fill_data(T *Parameter_0_0_host, T *Parameter_1_0_host, bool random = true);
    /**
     * @brief Prepare memory and data of the input and output
     */
    template <typename T>
    void prepare_tensor_template(T **Parameter_0_0, T **Parameter_1_0, T **Result_3_0, T **Parameter_0_0_host,
                                 T **Parameter_1_0_host, bool random = true);
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor(bool random = true) {}
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {}
    /**
     * @brief Transpose the colomn-order stored matrix
     */
    template <typename T> T *transpose(const T *matrix, int m, int n, int batch_count);
    /**
     * @brief Matrix multiply calculation on CPU side with input data and output data
     */
    template <typename T1, typename T2>
    void matrix_calculation_on_cpu_with_data(const T1 *Parameter_0_0_host, const T1 *Parameter_1_0_host,
                                             const T1 *Result_3_0, T2 **Result_cpu, T2 alpha = 1, T2 beta = 0);
    /**
     * @brief Check if the error < eps between the calculation result of GPU and CPU for each element in the matrix
     */
    template <typename T1, typename T2>
    int check_result(int batch_count, T1 *Result_3_0, T2 *Result_cpu, double eps = 1.e-6);
    /**
     * @brief Virtual function of Matrix multiply calculation on CPU side
     */
    virtual void matrix_calculation_on_cpu() {}
    /**
     * @brief Virtual function of Check the cublas function calculation correctness
     */
    virtual int correctness_check() { return 0; }

  public:
    /**
     * @brief Set the num test member
     * @param  num_test     the number of steps used to test and measure
     */
    void set_num_test(int num_test) { this->num_test = num_test; }
    /**
     * @brief Set the warm up member
     * @param  warm_up     the number of steps used to warm up
     */
    void set_warm_up(int warm_up) { this->warm_up = warm_up; }
    /**
     * @brief Set the num in step member
     * @param  num_in_step      the number of function invoking in a step
     */
    void set_num_in_step(int num_in_step) { this->num_in_step = num_in_step; }
    /**
     * @brief Set the random seed
     * @param  random_seed      random seed
     */
    void set_random_seed(int random_seed) { this->random_seed = random_seed; }
    /**
     * @brief Set the correctness
     * @param  correctness_check      if check the correctness of the function result
     */
    void set_correctness(int correctness_check) { this->correctness = correctness_check; }
    /**
     * @brief Set the eps
     * @param  eps      the acceptable error bound for numeric stability
     */
    void set_eps(double eps) { this->eps = eps; }
    /**
     * @brief Set the random data
     * @param  random_data      if generate random data
     */
    void set_random_data(bool random_data) { this->random_data = random_data; }
    /**
     * @brief Set the params string
     * @param  str             the str representing the params of the function
     */
    void set_function(std::string &str) { this->function_str_ = str; }
    /**
     * @brief Set the name member
     * @param  name             the name of the cublas function
     */
    void set_name(std::string &name) { this->name_ = name; }
    /**
     * @brief Set the m
     * @param  m                the m dim of matrix
     */
    void set_m(int m) { this->m_ = m; }
    /**
     * @brief Set the n
     * @param  n                the n dim of matrix
     */
    void set_n(int n) { this->n_ = n; }
    /**
     * @brief Set the k
     * @param  k                the k dim of matrix
     */
    void set_k(int k) { this->k_ = k; }
    /**
     * @brief Set the transa
     * @param  transa           whether the first matrix transpose
     */
    void set_transa(int transa) { this->transa_ = transa; }
    /**
     * @brief Set the transb
     * @param  transb           whether the second matrix transpose
     */
    void set_transb(int transb) { this->transb_ = transb; }
    /**
     * @brief Set the datatype
     * @param  datatype         data type used in cublasGemmEx and cublasGemmStridedBatchedEx
     */
    void set_datatype(std::string datatype) { this->datatype_ = datatype; }
    /**
     * @brief Set the use_tensor_core
     * @param  use_tensor_core  choose the algo used in cublasGemmEx and cublasGemmStridedBatchedEx
     */
    void set_use_tensor_core(bool use_tensor_core) { this->use_tensor_core_ = use_tensor_core; }
    /**
     * @brief Set the batch count
     * @param  batch_count      the num of the batch
     */
    void set_batch_count(int batch_count) { this->batch_count_ = batch_count; }
    /**
     * @brief Get the e name
     * @return cublas_function_name_enum
     */
    cublas_function_name_enum get_e_name() { return e_name_; }
    /**
     * @brief Get the name object
     * @return std::string name of the function
     */
    std::string get_name() { return this->name_; }
    /**
     * @brief   Convert function name to enum type
     * @return cublas_function_name_enum
     */
    cublas_function_name_enum name2enum() {
        auto it = cublas_function_name_string.find(this->name_);
        if (it != cublas_function_name_string.end()) {
            this->e_name_ = it->second;
            return e_name_;
        } else {
            throw "invalid input function name";
        }
    }
    /**
     * @brief The main procedure for cublas function test, includingwarmup, function test, time measurement
     * and output raw data results
     */
    void benchmark();
    /**
     * @brief Destroy the Cublas Function object
     */
    virtual ~CublasFunction() {}
};

/**
 * @brief Fill the random data into the input in float type
 */
template <> void CublasFunction::fill_data(float *Parameter_0_0_host, float *Parameter_1_0_host, bool random) {
    if (random) {
        srand(random_seed);
        for (int i = 0; i < m_ * k_ * batch_count_; i++) {
            Parameter_0_0_host[i] = ((float)rand() / (float)(RAND_MAX));
        }
        for (int i = 0; i < k_ * n_ * batch_count_; ++i) {
            Parameter_1_0_host[i] = ((float)rand() / (float)(RAND_MAX));
        }
    } else {
        // memset the input data to fixed float value
        memset(Parameter_0_0_host, 2,
               (unsigned long)m_ * (unsigned long)k_ * (unsigned long)batch_count_ * sizeof(float));
        memset(Parameter_1_0_host, 3,
               (unsigned long)k_ * (unsigned long)n_ * (unsigned long)batch_count_ * sizeof(float));
    }
}
/**
 * @brief Fill the random data into the input in half type
 */
template <> void CublasFunction::fill_data(half *Parameter_0_0_host, half *Parameter_1_0_host, bool random) {
    if (random) {
        srand(random_seed);
        for (int i = 0; i < m_ * k_ * batch_count_; i++) {
            Parameter_0_0_host[i] = half((float)rand() / (float)(RAND_MAX));
        }
        for (int i = 0; i < k_ * n_ * batch_count_; ++i) {
            Parameter_1_0_host[i] = half((float)rand() / (float)(RAND_MAX));
        }
    } else {
        // memset the input data to fixed float value
        std::fill(Parameter_0_0_host, Parameter_0_0_host + m_ * k_ * batch_count_, half(2.0));
        std::fill(Parameter_1_0_host, Parameter_1_0_host + k_ * n_ * batch_count_, half(3.0));
    }
}
/**
 * @brief Fill the random data into the input in cuComplex type
 */
template <> void CublasFunction::fill_data(cuComplex *Parameter_0_0_host, cuComplex *Parameter_1_0_host, bool random) {
    if (random) {
        srand(random_seed);
        for (int i = 0; i < m_ * k_ * batch_count_; i++) {
            Parameter_0_0_host[i] =
                make_cuComplex(((float)rand() / (float)(RAND_MAX)), ((float)rand() / (float)(RAND_MAX)));
        }
        for (int i = 0; i < k_ * n_ * batch_count_; ++i) {
            Parameter_1_0_host[i] =
                make_cuComplex(((float)rand() / (float)(RAND_MAX)), ((float)rand() / (float)(RAND_MAX)));
        }
    } else {
        // memset the input data to fixed float value
        std::fill(Parameter_0_0_host, Parameter_0_0_host + m_ * k_ * batch_count_, make_cuComplex(2.0f, 2.0f));
        std::fill(Parameter_1_0_host, Parameter_1_0_host + k_ * n_ * batch_count_, make_cuComplex(3.0f, 3.0f));
    }
}
/**
 * @brief Prepare memory and data of the input and output
 */
template <typename T>
void CublasFunction::prepare_tensor_template(T **Parameter_0_0, T **Parameter_1_0, T **Result_3_0,
                                             T **Parameter_0_0_host, T **Parameter_1_0_host, bool random) {
    int m = this->m_, n = this->n_, k = this->k_, batch_count = this->batch_count_;
    // input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)Parameter_0_0_host, sizeof(T) * m * k * batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_0_0, sizeof(T) * m * k * batch_count_));
    // input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)Parameter_1_0_host, sizeof(T) * n * k * batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_1_0, sizeof(T) * n * k * batch_count_));

    // fill input values
    fill_data(reinterpret_cast<T *>(*Parameter_0_0_host), reinterpret_cast<T *>(*Parameter_1_0_host), random);

    // copy input data from host to device
    CUDA_SAFE_CALL(
        cudaMemcpy(*Parameter_0_0, *Parameter_0_0_host, sizeof(T) * m * k * batch_count_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(*Parameter_1_0, *Parameter_1_0_host, sizeof(T) * k * n * batch_count_, cudaMemcpyHostToDevice));

    // output arguments
    CUDA_SAFE_CALL(cudaMalloc((void **)Result_3_0, sizeof(T) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMemset((void *)*Result_3_0, 0, sizeof(T) * m * n * batch_count_));
}
/**
 * @brief Transpose the colomn-order stored matrix with float or half datatype
 */
template <typename T> T *CublasFunction::transpose(const T *matrix, int m, int n, int batch_count) {
    T *transpose_matrix = (T *)malloc((unsigned long)m * (unsigned long)n * sizeof(T) * (unsigned long)batch_count);
    for (int b = 0; b < batch_count; b++) {
        for (int i = 0; i < m * n; i++) {
            int c = i / m;
            int r = i % m;
            int tran_i = r * n + c;
            transpose_matrix[tran_i + b * m * n] = matrix[i + b * m * n];
        }
    }
    return transpose_matrix;
}
/**
 * @brief Matrix multiply calculation on CPU side
 */
template <typename T1, typename T2>
void CublasFunction::matrix_calculation_on_cpu_with_data(const T1 *Parameter_0_0_host, const T1 *Parameter_1_0_host,
                                                         const T1 *Result_3_0, T2 **Result_cpu, T2 alpha, T2 beta) {
    int m = this->m_, n = this->n_, k = this->k_, batch_count = this->batch_count_;
    // Copy result from device to host
    T1 *Result_3_0_host;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Result_3_0_host,
                                  sizeof(T1) * (unsigned long)m * (unsigned long)n * (unsigned long)batch_count));
    CUDA_SAFE_CALL(cudaMemcpy(Result_3_0_host, Result_3_0,
                              sizeof(T1) * (unsigned long)m * (unsigned long)n * (unsigned long)batch_count,
                              cudaMemcpyDeviceToHost));
    // Transpose the input matrix
    T1 *Parameter_0_0_host_op, *Parameter_1_0_host_op;
    Parameter_0_0_host_op = (T1 *)malloc((unsigned long)m * (unsigned long)k * sizeof(T1) * (unsigned long)batch_count);
    Parameter_1_0_host_op = (T1 *)malloc((unsigned long)n * (unsigned long)k * sizeof(T1) * (unsigned long)batch_count);
    memcpy(Parameter_0_0_host_op, Parameter_0_0_host,
           (unsigned long)m * (unsigned long)k * sizeof(T1) * (unsigned long)batch_count);
    memcpy(Parameter_1_0_host_op, Parameter_1_0_host,
           (unsigned long)n * (unsigned long)k * sizeof(T1) * (unsigned long)batch_count);
    if (this->transa_) {
        Parameter_0_0_host_op = transpose(Parameter_0_0_host, k, m, batch_count);
    }
    if (this->transb_) {
        Parameter_1_0_host_op = transpose(Parameter_1_0_host, n, k, batch_count);
    }
    // C + i*strideC = alpha*op(A+i*strideA)*op(B+i*strideB)+beta(C+i*strideC), for i in [0, batchcount -1 ]
    // reference in https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference
    *Result_cpu = (T2 *)malloc((unsigned long)m * (unsigned long)n * sizeof(T2) * (unsigned long)batch_count);
    for (int b = 0; b < batch_count; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                (*Result_cpu)[i + j * m + b * m * n] = beta * (T2)(Result_3_0_host[i + j * m + b * m * n]);
                for (int p = 0; p < k; p++) {
                    (*Result_cpu)[i + j * m + b * m * n] += (T2)(Parameter_0_0_host_op[p * m + i + b * m * k] *
                                                                 Parameter_1_0_host_op[j * k + p + b * k * n]);
                    (*Result_cpu)[i + j * m + b * m * n] *= alpha;
                }
            }
        }
    }
    CUDA_SAFE_CALL(cudaFreeHost(Result_3_0_host));
    free(Parameter_0_0_host_op);
    free(Parameter_1_0_host_op);
}
/**
 * @brief Transpose the colomn-order stored matrix with complex datatype
 */
template <>
void CublasFunction::matrix_calculation_on_cpu_with_data(const cuComplex *Parameter_0_0_host,
                                                         const cuComplex *Parameter_1_0_host,
                                                         const cuComplex *Result_3_0, std::complex<float> **Result_cpu,
                                                         std::complex<float> alpha, std::complex<float> beta) {
    int m = this->m_, n = this->n_, k = this->k_, batch_count = this->batch_count_;
    // Copy result from device to host
    std::complex<float> *Result_3_0_host;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Result_3_0_host, sizeof(std::complex<float>) * m * n * batch_count));
    CUDA_SAFE_CALL(cudaMemcpy(Result_3_0_host, Result_3_0, sizeof(std::complex<float>) * m * n * batch_count,
                              cudaMemcpyDeviceToHost));
    cuComplex *Parameter_0_0_host_op, *Parameter_1_0_host_op;
    Parameter_0_0_host_op =
        (cuComplex *)malloc((unsigned long)m * (unsigned long)k * sizeof(cuComplex) * (unsigned long)batch_count);
    Parameter_1_0_host_op =
        (cuComplex *)malloc((unsigned long)n * (unsigned long)k * sizeof(cuComplex) * (unsigned long)batch_count);
    memcpy(Parameter_0_0_host_op, Parameter_0_0_host,
           (unsigned long)m * (unsigned long)k * sizeof(cuComplex) * (unsigned long)batch_count);
    memcpy(Parameter_1_0_host_op, Parameter_1_0_host,
           (unsigned long)n * (unsigned long)k * sizeof(cuComplex) * (unsigned long)batch_count);
    if (this->transa_) {
        Parameter_0_0_host_op = transpose<cuComplex>(Parameter_0_0_host, k, m, batch_count);
    }
    if (this->transb_) {
        Parameter_1_0_host_op = transpose<cuComplex>(Parameter_1_0_host, n, k, batch_count);
    }

    *Result_cpu = (std::complex<float> *)malloc((unsigned long)m * (unsigned long)n * sizeof(std::complex<float>) *
                                                (unsigned long)batch_count);

    for (int b = 0; b < batch_count; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                (*Result_cpu)[i + j * m + b * m * n] = beta * Result_3_0_host[i + j * m + b * m * n];
                for (int p = 0; p < k; p++) {
                    (*Result_cpu)[i + j * m + b * m * n] +=
                        std::complex<float>(Parameter_0_0_host_op[p * m + i + b * m * k].x,
                                            Parameter_0_0_host_op[p * m + i + b * m * k].y) *
                        std::complex<float>(Parameter_1_0_host_op[j * k + p + b * k * n].x,
                                            Parameter_1_0_host_op[j * k + p + b * k * n].y);
                    (*Result_cpu)[i + j * m + b * m * n] *= alpha;
                }
            }
        }
    }
    CUDA_SAFE_CALL(cudaFreeHost(Result_3_0_host));
    free(Parameter_0_0_host_op);
    free(Parameter_1_0_host_op);
}
/**
 * @brief Check if the error < eps between the calculation result of GPU and CPU for each element in the matrix
 */
template <typename T1, typename T2>
int CublasFunction::check_result(int batch_count, T1 *Result_3_0, T2 *Result_cpu, double eps) {
    int m = this->m_, n = this->n_, k = this->k_;
    // Copy result from device to host
    T1 *Result_3_0_host;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Result_3_0_host, sizeof(T1) * m * n * batch_count));
    CUDA_SAFE_CALL(cudaMemcpy(Result_3_0_host, Result_3_0, sizeof(T1) * m * n * batch_count, cudaMemcpyDeviceToHost));

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/|<x, y>_cpu|/dot_length  < eps
    int error_count = 0;
    for (int i = 0; i < static_cast<int>(m * n) * batch_count; i++) {
        double abs_err = fabs(Result_cpu[i] - (T2)(Result_3_0_host[i]));
        double dot_length = k;
        double abs_val = fabs(Result_cpu[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("error! matrix[%05d]=%.8f, ref=%.8f error term %.8f is > %E\n", i, (float)Result_3_0_host[i],
                   Result_cpu[i], rel_err, eps);
            error_count += 1;
        }
    }
    CUDA_SAFE_CALL(cudaFreeHost(Result_3_0_host));
    free(Result_cpu);
    return error_count;
}
/**
 * @brief Check if the error < eps between the calculation result of GPU and CPU for each element in the matrix
 */
template <>
int CublasFunction::check_result(int batch_count, cuComplex *Result_3_0, std::complex<float> *Result_cpu, double eps) {
    int m = this->m_, n = this->n_, k = this->k_;
    // Copy result from device to host
    std::complex<float> *Result_3_0_host;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Result_3_0_host, sizeof(std::complex<float>) * m * n * batch_count));
    CUDA_SAFE_CALL(cudaMemcpy(Result_3_0_host, Result_3_0, sizeof(std::complex<float>) * m * n * batch_count,
                              cudaMemcpyDeviceToHost));
    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/|<x, y>_cpu|/dot_length  < eps
    int error_count = 0;
    for (int i = 0; i < static_cast<int>(m * n) * batch_count; i++) {
        double abs_err = fabs(Result_cpu[i] - Result_3_0_host[i]);
        double dot_length = k;
        double abs_val = fabs(Result_cpu[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("error! matrix[%05d]=%.8f,%.8f, ref=%.8f,%.8f error term %.8f is > %E\n", i,
                   Result_3_0_host[i].real(), Result_3_0_host[i].imag(), Result_cpu[i].real(), Result_cpu[i].imag(),
                   rel_err, eps);
            error_count += 1;
        }
    }
    CUDA_SAFE_CALL(cudaFreeHost(Result_3_0_host));
    free(Result_cpu);
    return error_count;
}
/**
 * @brief The main procedure for cublas function test, including warmup, function test, time measurement and output raw
 * data results
 */
void CublasFunction::benchmark() {
    // Malloc memory for input and output data
    bool random = this->correctness ? true : this->random_data;
    this->prepare_tensor(random);

    // Warm up
    for (int i_ = 0; i_ < warm_up; i_++) {
        this->kernel_entry();
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Prepare some varibles for time measurement
    std::vector<float> iteration_time;
    int errors = 0;
    // Benchmark in range of steps
    for (int i_ = 0; i_ < num_test; i_++) {
        // Collect time within each step, including #repeat_in_one_step times function invoking
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < num_in_step; j++) {
            if (this->correctness)
                this->matrix_calculation_on_cpu();
            this->kernel_entry();
            if (this->correctness) {
                errors += this->correctness_check();
            }
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        // Convert step time to single function duration and update min and max duration
        float i = static_cast<float>(std::chrono::duration<double, std::micro>(end - start).count() / num_in_step);
        iteration_time.emplace_back(i);
    }

    // Output results
    std::cout << "[function config]: " << this->function_str_ << std::endl;
    std::cout << "[raw_data]: ";
    for (int i = 0; i < iteration_time.size(); i++) {
        std::cout << iteration_time[i] << ",";
    }
    std::cout << std::endl;
    if (this->correctness) {
        std::string correctness_str = errors == 0 ? "Result = PASS" : "Result = FAIL";
        std::cout << "[correctness]: " << correctness_str
                  << ", error rate: " << errors / (num_in_step * num_test * this->m_ * this->n_) << std::endl;
    }
}
