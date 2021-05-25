/**
 * @copyright Copyright (c) Microsoft Corporation
 * @file cublas_benchmark.h
 * @brief Unify a base class for cublas function benchmark
 */

#pragma once

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "cublas_helper.h"

enum cublas_function_name_enum {
    e_cublasSgemm = 0,
    e_cublasCgemm,
    e_cublasGemmEx,
    e_cublasGemmStridedBatchedEx,
    e_cublasSgemmStridedBatched,
    e_cublasCgemm3mStridedBatched
};

static std::unordered_map<std::string, cublas_function_name_enum> const cublas_function_name_string = {
    {"cublasSgemm", cublas_function_name_enum::e_cublasSgemm},
    {"cublasCgemm", cublas_function_name_enum::e_cublasCgemm},
    {"cublasGemmEx", cublas_function_name_enum::e_cublasGemmEx},
    {"cublasGemmStridedBatchedEx", cublas_function_name_enum::e_cublasGemmStridedBatchedEx},
    {"cublasSgemmStridedBatched", cublas_function_name_enum::e_cublasSgemmStridedBatched},
    {"cublasCgemm3mStridedBatched", cublas_function_name_enum::e_cublasCgemm3mStridedBatched},
};

// Class to store params of cublas function and run the benchmark of this function
class CublasFunction {
  protected:
    int num_test;
    int warm_up;
    int num_in_step;
    std::string name_;
    int m_;
    int k_;
    int n_;
    int transa_;
    int transb_;
    std::string datatype_;
    bool use_tensor_core_;
    int batch_count_ = 1;
    cublas_function_name_enum e_name_;
    std::string to_str_;
    cublasHandle_t cublas_handle;

  public:
    void set_num_test(int num_test) { this->num_test = num_test; }
    void set_warm_up(int warm_up) { this->warm_up = warm_up; }
    void set_num_in_step(int num_in_step) { this->num_in_step = num_in_step; }
    void set_str(std::string &str) { this->to_str_ = str; }
    void set_name(std::string &name) { this->name_ = name; }
    void set_m(int m) { this->m_ = m; }
    void set_n(int n) { this->n_ = n; }
    void set_k(int k) { this->k_ = k; }
    void set_transa(int transa) { this->transa_ = transa; }
    void set_transb(int transb) { this->transb_ = transb; }
    void set_datatype(std::string datatype) { this->datatype_ = datatype; }
    void set_use_tensor_core(bool use_tensor_core) { this->use_tensor_core_ = use_tensor_core; }
    void set_batch_count(int batch_count) { this->batch_count_ = batch_count; }
    cublas_function_name_enum get_e_name() { return e_name_; }
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
    /**
     * @brief The main procedure for cublas function test, including cuda init, warmup, function test, time measurement
     * and cuda free
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

    float *Parameter_0_0_host, *Parameter_1_0_host, *Result_3_0_host;
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
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Result_3_0_host, sizeof(float) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Result_3_0, sizeof(float) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMemset((void *)*Result_3_0, 0, sizeof(float) * m * n * batch_count_));

    CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Result_3_0_host));
}
/**
 * @brief Prepare memory and data of the input and output in cuComplex type
 */
void CublasFunction::prepare_tensor_cucomplex(cuComplex **Parameter_0_0, cuComplex **Parameter_1_0,
                                              cuComplex **Result_3_0) {
    int m = this->m_;
    int n = this->n_;
    int k = this->k_;

    cuComplex *Parameter_0_0_host, *Parameter_1_0_host, *Result_3_0_host;
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
    CUDA_SAFE_CALL(cudaMallocHost((void **)&Result_3_0_host, sizeof(cuComplex) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Result_3_0, sizeof(cuComplex) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMemset((void *)*Result_3_0, 0, sizeof(cuComplex) * m * n * batch_count_));

    CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Result_3_0_host));
}
/**
 * @brief The main procedure for cublas function test, including cuda init, warmup, function test, time measurement
 * and cuda free
 */
void CublasFunction::benchmark() {
    // Init cuda handle and set device
    cuda_init(&cublas_handle);

    // Malloc memory for input and output data
    this->prepare_tensor();

    // Warm up
    for (int i_ = 0; i_ < warm_up; i_++) {
        this->kernel_entry();
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Prepare some varibles for time measurement
    std::vector<float> iteration_time;
    // Benchmark in range of steps
    int repeat_in_one_step = num_in_step;
    for (int i_ = 0; i_ < num_test; i_++) {
        // Collect time within each step, including #repeat_in_one_step times function invoking
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < repeat_in_one_step; j++) {
            this->kernel_entry();
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        // Convert step time to single function duration and update min and max duration
        float i =
            static_cast<float>(std::chrono::duration<double, std::micro>(end - start).count() / repeat_in_one_step);
        iteration_time.emplace_back(i);
    }

    // Output results
    std::cout << "[function config]: " << this->to_str_ << std::endl;
    std::cout << "[raw_data]: ";
    for (int i = 0; i < iteration_time.size(); i++) {
        std::cout << iteration_time[i] << ",";
    }
    std::cout << std::endl;

    cuda_free(&cublas_handle);
}
