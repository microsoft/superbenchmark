// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include <time.h>
#include <chrono>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuComplex.h>

#include "cmd_helper.h"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result != cudaSuccess)
    {
        const char *msg = cudaGetErrorString(result);
        std::stringstream safe_call_ss;
        safe_call_ss << "\nerror: " << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << msg;
        // Make sure we call CUDA Device Reset before exiting
        throw std::runtime_error(safe_call_ss.str());
    }
}
#define CUDA_SAFE_CALL(x) check_cuda((x), #x, __FILE__, __LINE__)

void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line)
{
    if (result != CUBLAS_STATUS_SUCCESS)
    {

        std::stringstream safe_call_ss;
        safe_call_ss << "\nerror: " << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << result;
        // Make sure we call CUDA Device Reset before exiting
        throw std::runtime_error(safe_call_ss.str());
    }
}
#define CUBLAS_SAFE_CALL(x) check_cublas((x), #x, __FILE__, __LINE__)

cublasHandle_t cublas_handle;

enum cublas_function_name_enum
{
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

//Cuda context init
void cuda_init();
//Cuda context free
void cuda_free();

//Class to store params of cublas function and run the benchmark
class CublasFunction
{
private:
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

public:
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

    // Prepare input and output tensor for kernel running
    template <class T>
    void prepare_tensor(T **Parameter_0_0, T **Parameter_1_0, T **Result_3_0, T **Parameter_0_0_host, T **Parameter_1_0_host, T **Result_3_0_host);

    // Fill the random data into the input tensor
    template <class T>
    void fill_data(T *Parameter_0_0_host, T *Parameter_1_0_host);

    // Execute the kernel/function
    template <class T>
    int kernel_entry(T *a, T *b, T *c);

    // Convert function name to enum type
    cublas_function_name_enum name2enum()
    {
        auto it = cublas_function_name_string.find(this->name_);
        if (it != cublas_function_name_string.end())
        {
            this->e_name_ = it->second;
            return e_name_;
        }
        else
        {
            std::cout << "invalid input function name";
            exit(0);
        }
    }

    // The main procedure for cublas function test, including cuda init, warmup, function test, time measurement and cuda free
    template <class T>
    void benchmark(Options *options);

    // Wrappers of cublas functions
    template <class T>
    void gemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const T *a, const T *b, T *c);

    void gemmEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *A, const void *B, void *C, std::string type, bool use_tensor_core);

    void gemmStridedBatchedEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *a, const void *b, void *c, std::string type, bool use_tensor_core, int batchCount);

    void Cgemm3mStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a, const cuComplex *b, cuComplex *c, int batchCount);

    void SgemmStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a, const float *b, float *c, int batchCount);
};

template <class T>
void CublasFunction::fill_data(T *Parameter_0_0_host, T *Parameter_1_0_host)
{
    std::cout << "invalid type";
    exit(0);
}

template <>
void CublasFunction::fill_data<float>(float *Parameter_0_0_host, float *Parameter_1_0_host)
{
    for (int i = 0; i < m_ * k_; i++)
    {
        Parameter_0_0_host[i] = (float)rand() / (float)(RAND_MAX);
    }
    for (int i = 0; i < k_ * n_; ++i)
    {
        Parameter_1_0_host[i] = (float)rand() / (float)(RAND_MAX);
    }
}

template <>
void CublasFunction::fill_data<cuComplex>(cuComplex *Parameter_0_0_host, cuComplex *Parameter_1_0_host)
{
    for (int i = 0; i < m_ * k_; i++)
    {
        Parameter_0_0_host[i] = make_cuComplex(((float)rand() / (float)(RAND_MAX)), ((float)rand() / (float)(RAND_MAX)));
    }
    for (int i = 0; i < k_ * n_; ++i)
    {
        Parameter_1_0_host[i] = make_cuComplex(((float)rand() / (float)(RAND_MAX)), ((float)rand() / (float)(RAND_MAX)));
    }
}

template <class T>
void CublasFunction::prepare_tensor(T **Parameter_0_0, T **Parameter_1_0, T **Result_3_0, T **Parameter_0_0_host, T **Parameter_1_0_host, T **Result_3_0_host)
{
    int m = this->m_;
    int n = this->n_;
    int k = this->k_;

    //input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)Parameter_0_0_host, sizeof(T) * m * k * this->batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_0_0, sizeof(T) * m * k * this->batch_count_));
    //input argument
    CUDA_SAFE_CALL(cudaMallocHost((void **)Parameter_1_0_host, sizeof(T) * n * k * this->batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Parameter_1_0, sizeof(T) * n * k * this->batch_count_));

    // fill input values
    fill_data<T>(*Parameter_0_0_host, *Parameter_1_0_host);

    // copy input data from host to device
    CUDA_SAFE_CALL(cudaMemcpy(*Parameter_0_0, *Parameter_0_0_host, sizeof(T) * m * k * this->batch_count_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(*Parameter_1_0, *Parameter_1_0_host, sizeof(T) * k * n * this->batch_count_, cudaMemcpyHostToDevice));

    //output arguments
    CUDA_SAFE_CALL(cudaMallocHost((void **)Result_3_0_host, sizeof(T) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMalloc((void **)Result_3_0, sizeof(T) * m * n * batch_count_));
    CUDA_SAFE_CALL(cudaMemset((void *)*Result_3_0, 0, sizeof(T) * m * n * batch_count_));
}

template <class T>
int CublasFunction::kernel_entry(T *a, T *b, T *c)
{
    switch (this->e_name_)
    {
    case e_cublasSgemm:
        gemm<float>(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
                    reinterpret_cast<const float *>(a),
                    reinterpret_cast<const float *>(b),
                    reinterpret_cast<float *>(c));

        break;
    case e_cublasCgemm:
        gemm<cuComplex>(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
                        reinterpret_cast<const cuComplex *>(a),
                        reinterpret_cast<const cuComplex *>(b),
                        reinterpret_cast<cuComplex *>(c));
        break;
    case e_cublasGemmEx:
        gemmEx(cublas_handle, this->transa_, this->transb_,
               this->m_, this->n_, this->k_,
               reinterpret_cast<void *>(a), reinterpret_cast<void *>(b), reinterpret_cast<void *>(c),
               this->datatype_, this->use_tensor_core_);
        break;
    case e_cublasGemmStridedBatchedEx:
        gemmStridedBatchedEx(cublas_handle,
                             this->transa_, this->transb_, this->m_, this->n_, this->k_,
                             reinterpret_cast<void *>(a), reinterpret_cast<void *>(b), reinterpret_cast<void *>(c),
                             this->datatype_, this->use_tensor_core_, this->batch_count_);
        break;
    case e_cublasSgemmStridedBatched:
        SgemmStridedBatched(cublas_handle,
                            this->transa_, this->transb_, this->m_, this->n_, this->k_,
                            reinterpret_cast<const float *>(a),
                            reinterpret_cast<const float *>(b),
                            reinterpret_cast<float *>(c),
                            this->batch_count_);
        break;
    case e_cublasCgemm3mStridedBatched:
        Cgemm3mStridedBatched(cublas_handle,
                              this->transa_, this->transb_, this->m_, this->n_, this->k_,
                              reinterpret_cast<const cuComplex *>(a),
                              reinterpret_cast<const cuComplex *>(b),
                              reinterpret_cast<cuComplex *>(c), this->batch_count_);
        break;
    default:
        std::cout << "invalid enum name";
        exit(0);
    }
    return 0;
}

template <class T>
void CublasFunction::benchmark(Options *options)
{
    // Tet warmup and test step nums parsed from cmd
    int steps = options->num_test;
    int warm_up = options->warm_up;

    // Init cuda handle and set device
    cuda_init();

    // Malloc memory for input and output data
    T *Parameter_0_0, *Parameter_1_0, *Result_3_0, *Parameter_0_0_host, *Parameter_1_0_host, *Result_3_0_host;
    this->prepare_tensor<T>(&Parameter_0_0, &Parameter_1_0, &Result_3_0, &Parameter_0_0_host, &Parameter_1_0_host, &Result_3_0_host);

    // Warm up
    for (int i_ = 0; i_ < warm_up; i_++)
    {
        this->kernel_entry<T>(Parameter_0_0, Parameter_1_0, Result_3_0);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Prepare some varibles for time measurement
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_total = 0;
    std::vector<float> iteration_time;

    // Benchmark in range of steps
    int repeat_in_one_step = options->num_in_step;
    for (int i_ = 0; i_ < steps; i_++)
    {
        // Collect time within each step, including #repeat_in_one_step times function invoking
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < repeat_in_one_step; j++)
        {
            this->kernel_entry<T>(Parameter_0_0, Parameter_1_0, Result_3_0);
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        // Convert step time to single function duration and update min and max duration
        float i = static_cast<float>(std::chrono::duration<double, std::micro>(end - start).count() / repeat_in_one_step);
        iteration_time.emplace_back(i);
        if (i > ms_max)
            ms_max = i;
        if (i < ms_min)
            ms_min = i;
        ms_total += i;
    }

    // Output results
    std::cout << "[function config]: " << this->to_str_ << std::endl;
    std::cout << "[min, max, mean]: "
              << "[" << ms_min << " " << ms_max << " " << ms_total / steps << "]"
              << std::endl;
    std::cout << "[raw_data]: ";
    for (int i = 0; i < iteration_time.size(); i++)
    {
        std::cout << iteration_time[i] << ",";
    }
    std::cout << std::endl;

    //Free context
    CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
    CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
    CUDA_SAFE_CALL(cudaFree(Result_3_0));
    CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
    CUDA_SAFE_CALL(cudaFreeHost(Result_3_0_host));
    cuda_free();
}

// Helper function to convert json to cublasfunction
void from_json(const json &j, CublasFunction &fn)
{
    auto str = j.dump();
    std::replace(str.begin(), str.end(), '\"', ' ');
    fn.set_str(str);
    auto name = j.at("name").get<std::string>();
    fn.set_name(name);
    auto m = j.at("m").get<int>();
    fn.set_m(m);
    auto n = j.at("n").get<int>();
    fn.set_n(n);
    auto k = j.at("k").get<int>();
    fn.set_k(k);
    auto transa = j.at("transa").get<int>();
    fn.set_transa(transa);
    auto transb = j.at("transb").get<int>();
    fn.set_transb(transb);
    fn.name2enum();
    switch (fn.get_e_name())
    {
    case e_cublasSgemmStridedBatched:
    {
        auto batch_count = j.at("batchCount").get<int>();
        fn.set_batch_count(batch_count);
        break;
    }
    case e_cublasGemmStridedBatchedEx:
    {
        auto batch_count = j.at("batchCount").get<int>();
        fn.set_batch_count(batch_count);
    }
    case e_cublasGemmEx:
    {
        auto datatype = j.at("datatype").get<std::string>();
        fn.set_datatype(datatype);
        auto use_tensor_core = j.at("use_tensor_core").get<bool>();
        fn.set_use_tensor_core(use_tensor_core);
        break;
    }
    case e_cublasCgemm3mStridedBatched:
    {
        auto batch_count = j.at("batchCount").get<int>();
        fn.set_batch_count(batch_count);
        break;
    }
    case e_cublasSgemm:
    case e_cublasCgemm:
    {
        fn.set_batch_count(1);
        break;
    }
    default:
        std::cout << "invalid function name";
        exit(-1);
    }
}
