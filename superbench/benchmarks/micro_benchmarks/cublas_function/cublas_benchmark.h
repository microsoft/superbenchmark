// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cublas_benchmark.h
 * @brief Unify a base class for cublas function benchmark
 */

#pragma once

#include <chrono>
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
     * @brief Fill the random data into the input in float type
     */
    void fill_data_float(float *Parameter_0_0_host, float *Parameter_1_0_host);
    /**
     * @brief Fill the random data into the input in cuComplex type
     */
    void fill_data_cucomplex(cuComplex *Parameter_0_0_host, cuComplex *Parameter_1_0_host);
    /**
     * @brief Prepare memory and data of the input and output in float type
     */
    void prepare_tensor_float(float **Parameter_0_0, float **Parameter_1_0, float **Result_3_0);
    /**
     * @brief Prepare memory and data of the input and output in cuComplex type
     */
    void prepare_tensor_cucomplex(cuComplex **Parameter_0_0, cuComplex **Parameter_1_0, cuComplex **Result_3_0);
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor() {}
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {}

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
 * @brief Fill the random data into the input in cuComplex type
 */
void CublasFunction::fill_data_float(float *Parameter_0_0_host, float *Parameter_1_0_host) {
    srand(random_seed);
    for (int i = 0; i < m_ * k_; i++) {
        Parameter_0_0_host[i] = (float)rand() / (float)(RAND_MAX);
    }
    for (int i = 0; i < k_ * n_; ++i) {
        Parameter_1_0_host[i] = (float)rand() / (float)(RAND_MAX);
    }
}
/**
 * @brief Fill the random data into the input in cuComplex type
 */
void CublasFunction::fill_data_cucomplex(cuComplex *Parameter_0_0_host, cuComplex *Parameter_1_0_host) {
    srand(random_seed);
    for (int i = 0; i < m_ * k_; i++) {
        Parameter_0_0_host[i] =
            make_cuComplex(((float)rand() / (float)(RAND_MAX)), ((float)rand() / (float)(RAND_MAX)));
    }
    for (int i = 0; i < k_ * n_; ++i) {
        Parameter_1_0_host[i] =
            make_cuComplex(((float)rand() / (float)(RAND_MAX)), ((float)rand() / (float)(RAND_MAX)));
    }
}
/**
 * @brief Prepare memory and data of the input and output in float type
 */
void CublasFunction::prepare_tensor_float(float **Parameter_0_0, float **Parameter_1_0, float **Result_3_0) {
    int m = this->m_;
    int n = this->n_;
    int k = this->k_;

    float *Parameter_0_0_host, *Parameter_1_0_host;
    // input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Parameter_0_0_host, sizeof(float) * m * k * this->batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_0_0, sizeof(float) * m * k * this->batch_count_));
    // input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Parameter_1_0_host, sizeof(float) * n * k * this->batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_1_0, sizeof(float) * n * k * this->batch_count_));

    // fill input values
    fill_data_float(Parameter_0_0_host, Parameter_1_0_host);

    // copy input data from host to device
    CUDA_SAFE_CALL(cudaMemcpy(*Parameter_0_0, Parameter_0_0_host, sizeof(float) * m * k * this->batch_count_,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(*Parameter_1_0, Parameter_1_0_host, sizeof(float) * k * n * this->batch_count_,
                              cudaMemcpyHostToDevice));

    // output arguments
    CUDA_SAFE_CALL(cudaMalloc((void **)Result_3_0, sizeof(float) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMemset((void *)*Result_3_0, 0, sizeof(float) * m * n * batch_count_));

    CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
}
/**
 * @brief Prepare memory and data of the input and output in cuComplex type
 */
void CublasFunction::prepare_tensor_cucomplex(cuComplex **Parameter_0_0, cuComplex **Parameter_1_0,
                                              cuComplex **Result_3_0) {
    int m = this->m_;
    int n = this->n_;
    int k = this->k_;

    cuComplex *Parameter_0_0_host, *Parameter_1_0_host;
    // input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Parameter_0_0_host, sizeof(cuComplex) * m * k * this->batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_0_0, sizeof(cuComplex) * m * k * this->batch_count_));
    // input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Parameter_1_0_host, sizeof(cuComplex) * n * k * this->batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_1_0, sizeof(cuComplex) * n * k * this->batch_count_));

    // fill input values
    fill_data_cucomplex(Parameter_0_0_host, Parameter_1_0_host);

    // copy input data from host to device
    CUDA_SAFE_CALL(cudaMemcpy(*Parameter_0_0, Parameter_0_0_host, sizeof(cuComplex) * m * k * this->batch_count_,
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(*Parameter_1_0, Parameter_1_0_host, sizeof(cuComplex) * k * n * this->batch_count_,
                              cudaMemcpyHostToDevice));

    // output arguments
    CUDA_SAFE_CALL(cudaMalloc((void **)Result_3_0, sizeof(cuComplex) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMemset((void *)*Result_3_0, 0, sizeof(cuComplex) * m * n * batch_count_));

    CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
}
/**
 * @brief The main procedure for cublas function test, including warmup, function test, time measurement and output raw
 * data results
 */
void CublasFunction::benchmark() {
    // Malloc memory for input and output data
    this->prepare_tensor();

    // Warm up
    for (int i_ = 0; i_ < warm_up; i_++) {
        for (int j = 0; j < num_in_step; j++) {
            this->kernel_entry();
        }
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Prepare some varibles for time measurement
    std::vector<float> iteration_time;
    // Benchmark in range of steps
    for (int i_ = 0; i_ < num_test; i_++) {
        // Collect time within each step, including #repeat_in_one_step times function invoking
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < num_in_step; j++) {
            this->kernel_entry();
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
}
