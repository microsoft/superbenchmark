// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cublas_function.h
 * @brief Implementation of specific cublas function
 */

#pragma once

#include "cublas_benchmark.h"

/**
 * @brief Class of SgemmFunction
 */
class SgemmFunction : public CublasFunction {
    float *Parameter_0_0;      ///< the pointer of the first input data
    float *Parameter_1_0;      ///< the pointer of the second input data
    float *Result_3_0;         ///< the pointer of output data
    float *Parameter_0_0_host; ///< the pointer of the first input data on host
    float *Parameter_1_0_host; ///< the pointer of the second input data on host
    float *Result_cpu;

    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        sgemm(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
              reinterpret_cast<const float *>(Parameter_0_0), reinterpret_cast<const float *>(Parameter_1_0),
              reinterpret_cast<float *>(Result_3_0));
    }
    /**
     * @brief  Function calculation on CPU side
     */
    virtual void matrix_calculation_on_cpu() {
        matrix_calculation_on_cpu_with_data(Parameter_0_0_host, Parameter_1_0_host, Result_3_0, &Result_cpu, 1.0f,
                                            1.0f);
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor(bool random) {
        prepare_tensor_template(&Parameter_0_0, &Parameter_1_0, &Result_3_0, &Parameter_0_0_host, &Parameter_1_0_host,
                                random);
    }
    /**
     * @brief Check the correctness of function calculation result
     */
    virtual int correctness_check() {
        double eps = this->eps == 0.0 ? 1.e-6 : this->eps;
        return check_result(1, Result_3_0, Result_cpu, eps);
    }

  public:
    /**
     * @brief Construct a new Sgemm Function object
     */
    SgemmFunction() {
        this->batch_count_ = 1;
        cuda_init(&cublas_handle);
    }
    /**
     * @brief Construct a new Sgemm Function object
     * @param  function         base class CublasFunction object
     */
    SgemmFunction(CublasFunction &function) : CublasFunction(function) {
        this->batch_count_ = 1;
        cuda_init(&cublas_handle);
    }
    /**
     * @brief Destroy the Sgemm Function object
     */
    ~SgemmFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
        cuda_free(&cublas_handle);
    }
};

/**
 * @brief Class of CgemmFunction
 */
class CgemmFunction : public CublasFunction {
    cuComplex *Parameter_0_0;
    cuComplex *Parameter_1_0;
    cuComplex *Result_3_0;
    cuComplex *Parameter_0_0_host;
    cuComplex *Parameter_1_0_host;
    std::complex<float> *Result_cpu;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        cgemm(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
              reinterpret_cast<const cuComplex *>(Parameter_0_0), reinterpret_cast<const cuComplex *>(Parameter_1_0),
              reinterpret_cast<cuComplex *>(Result_3_0));
    }
    /**
     * @brief  Function calculation on CPU side
     */
    virtual void matrix_calculation_on_cpu() {
        matrix_calculation_on_cpu_with_data(Parameter_0_0_host, Parameter_1_0_host, Result_3_0, &Result_cpu);
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor(bool random) {
        prepare_tensor_template(&Parameter_0_0, &Parameter_1_0, &Result_3_0, &Parameter_0_0_host, &Parameter_1_0_host,
                                random);
    }
    /**
     * @brief Check the correctness of function calculation result
     */
    virtual int correctness_check() {
        double eps = this->eps == 0.0 ? 1.e-6 : this->eps;
        return check_result(1, Result_3_0, Result_cpu, eps);
    }

  public:
    /**
     * @brief Construct a new Cgemm Function object
     */
    CgemmFunction() {
        this->batch_count_ = 1;
        cuda_init(&cublas_handle);
    }
    /**
     * @brief Construct a new Cgemm Function object
     * @param  function         base class CublasFunction object
     */
    CgemmFunction(CublasFunction &function) : CublasFunction(function) {
        this->batch_count_ = 1;
        cuda_init(&cublas_handle);
    }
    /**
     * @brief Destroy the Cgemm Function object
     */
    ~CgemmFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
        cuda_free(&cublas_handle);
    }
};

/**
 * @brief Class of GemmExFunction
 */
class GemmExFunction : public CublasFunction {
    void *Parameter_0_0;
    void *Parameter_1_0;
    void *Result_3_0;
    void *Parameter_0_0_host;
    void *Parameter_1_0_host;
    void *Result_cpu;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        gemmEx(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
               reinterpret_cast<void *>(Parameter_0_0), reinterpret_cast<void *>(Parameter_1_0),
               reinterpret_cast<void *>(Result_3_0), this->datatype_, this->use_tensor_core_);
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor(bool random) {
        if (this->datatype_.compare("half") == 0) {
            CublasFunction::prepare_tensor_template<half>(
                reinterpret_cast<half **>(&Parameter_0_0), reinterpret_cast<half **>(&Parameter_1_0),
                reinterpret_cast<half **>(&Result_3_0), reinterpret_cast<half **>(&Parameter_0_0_host),
                reinterpret_cast<half **>(&Parameter_1_0_host), random);
        } else if (this->datatype_.compare("float") == 0) {
            CublasFunction::prepare_tensor_template<float>(
                reinterpret_cast<float **>(&Parameter_0_0), reinterpret_cast<float **>(&Parameter_1_0),
                reinterpret_cast<float **>(&Result_3_0), reinterpret_cast<float **>(&Parameter_0_0_host),
                reinterpret_cast<float **>(&Parameter_1_0_host), random);
        }
    }
    /**
     * @brief  Function calculation on CPU side
     */
    virtual void matrix_calculation_on_cpu() {
        if (this->datatype_.compare("half") == 0) {
            matrix_calculation_on_cpu_with_data(
                reinterpret_cast<half *>(Parameter_0_0_host), reinterpret_cast<half *>(Parameter_1_0_host),
                reinterpret_cast<half *>(Result_3_0), reinterpret_cast<float **>(&Result_cpu));
        } else if (this->datatype_.compare("float") == 0) {
            matrix_calculation_on_cpu_with_data(
                reinterpret_cast<float *>(Parameter_0_0_host), reinterpret_cast<float *>(Parameter_1_0_host),
                reinterpret_cast<float *>(Result_3_0), reinterpret_cast<float **>(&Result_cpu));
        }
    }
    /**
     * @brief Check the correctness of function calculation result
     */
    virtual int correctness_check() {
        int result = 0;
        if (this->datatype_.compare("half") == 0) {
            double eps = this->eps == 0.0 ? 1.e-3 : this->eps;
            result = check_result(this->batch_count_, reinterpret_cast<half *>(Result_3_0),
                                  reinterpret_cast<float *>(Result_cpu), eps);
        } else if (this->datatype_.compare("float") == 0) {
            double eps = this->eps == 0.0 ? 1.e-6 : this->eps;
            result = check_result(this->batch_count_, reinterpret_cast<float *>(Result_3_0),
                                  reinterpret_cast<float *>(Result_cpu), eps);
        }
        return result;
    }

  public:
    /**
     * @brief Construct a new Gemm Ex Function object
     */
    GemmExFunction() {
        this->batch_count_ = 1;
        cuda_init(&cublas_handle);
    }
    /**
     * @brief Construct a new Gemm Ex Function object
     * @param  function         base class CublasFunction object
     */
    GemmExFunction(CublasFunction &function) : CublasFunction(function) {
        this->batch_count_ = 1;
        cuda_init(&cublas_handle);
    }
    /**
     * @brief Destroy the Gemm Ex Function object
     */
    ~GemmExFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
        cuda_free(&cublas_handle);
    }
};

/**
 * @brief Class of GemmStridedBatchedExFunction
 */
class GemmStridedBatchedExFunction : public CublasFunction {
    void *Parameter_0_0;
    void *Parameter_1_0;
    void *Result_3_0;
    void *Parameter_0_0_host;
    void *Parameter_1_0_host;
    void *Result_cpu;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        gemmStridedBatchedEx(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
                             reinterpret_cast<void *>(Parameter_0_0), reinterpret_cast<void *>(Parameter_1_0),
                             reinterpret_cast<void *>(Result_3_0), this->datatype_, this->use_tensor_core_,
                             this->batch_count_);
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor(bool random) {
        if (this->datatype_.compare("half") == 0) {
            prepare_tensor_template<half>(
                reinterpret_cast<half **>(&Parameter_0_0), reinterpret_cast<half **>(&Parameter_1_0),
                reinterpret_cast<half **>(&Result_3_0), reinterpret_cast<half **>(&Parameter_0_0_host),
                reinterpret_cast<half **>(&Parameter_1_0_host), random);
        } else if (this->datatype_.compare("float") == 0) {
            prepare_tensor_template<float>(
                reinterpret_cast<float **>(&Parameter_0_0), reinterpret_cast<float **>(&Parameter_1_0),
                reinterpret_cast<float **>(&Result_3_0), reinterpret_cast<float **>(&Parameter_0_0_host),
                reinterpret_cast<float **>(&Parameter_1_0_host), random);
        }
    }
    /**
     * @brief  Function calculation on CPU side
     */
    virtual void matrix_calculation_on_cpu() {
        if (this->datatype_.compare("half") == 0) {
            matrix_calculation_on_cpu_with_data(
                reinterpret_cast<half *>(Parameter_0_0_host), reinterpret_cast<half *>(Parameter_1_0_host),
                reinterpret_cast<half *>(Result_3_0), reinterpret_cast<float **>(&Result_cpu), 1.0f, 1.0f);
        } else if (this->datatype_.compare("float") == 0) {
            matrix_calculation_on_cpu_with_data(
                reinterpret_cast<float *>(Parameter_0_0_host), reinterpret_cast<float *>(Parameter_1_0_host),
                reinterpret_cast<float *>(Result_3_0), reinterpret_cast<float **>(&Result_cpu), 1.0f, 1.0f);
        }
    }
    /**
     * @brief Check the correctness of function calculation result
     */
    virtual int correctness_check() {
        int result = 0;
        if (this->datatype_.compare("half") == 0) {
            double eps = this->eps == 0.0 ? 1.e-3 : this->eps;
            result = check_result(this->batch_count_, reinterpret_cast<half *>(Result_3_0),
                                  reinterpret_cast<float *>(Result_cpu), eps);
        } else if (this->datatype_.compare("float") == 0) {
            double eps = this->eps == 0.0 ? 1.e-6 : this->eps;
            result = check_result(this->batch_count_, reinterpret_cast<float *>(Result_3_0),
                                  reinterpret_cast<float *>(Result_cpu), eps);
        }
        return result;
    }

  public:
    /**
     * @brief Construct a new Gemm Strided Batched Ex Function object
     */
    GemmStridedBatchedExFunction() { cuda_init(&cublas_handle); }
    /**
     * @brief Construct a new Gemm Strided Batched Ex Function object
     * @param  function         base class CublasFunction object
     */
    GemmStridedBatchedExFunction(CublasFunction &function) : CublasFunction(function) { cuda_init(&cublas_handle); }
    /**
     * @brief Destroy the Gemm Strided Batched Ex Function object
     */
    ~GemmStridedBatchedExFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
        cuda_free(&cublas_handle);
    }
};

/**
 * @brief Class of SgemmStridedBatchedFunction
 */
class SgemmStridedBatchedFunction : public CublasFunction {
    float *Parameter_0_0;
    float *Parameter_1_0;
    float *Result_3_0;
    float *Parameter_0_0_host;
    float *Parameter_1_0_host;
    float *Result_cpu;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        sgemmStridedBatched(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
                            reinterpret_cast<const float *>(Parameter_0_0),
                            reinterpret_cast<const float *>(Parameter_1_0), reinterpret_cast<float *>(Result_3_0),
                            this->batch_count_);
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor(bool random) {
        prepare_tensor_template(&Parameter_0_0, &Parameter_1_0, &Result_3_0, &Parameter_0_0_host, &Parameter_1_0_host,
                                random);
    }
    /**
     * @brief  Function calculation on CPU side
     */
    virtual void matrix_calculation_on_cpu() {
        matrix_calculation_on_cpu_with_data(Parameter_0_0_host, Parameter_1_0_host, Result_3_0, &Result_cpu, 1.0f,
                                            1.0f);
    }
    /**
     * @brief Check the correctness of function calculation result
     */
    virtual int correctness_check() {
        double eps = this->eps == 0.0 ? 1.e-6 : this->eps;
        return check_result(this->batch_count_, Result_3_0, Result_cpu, eps);
    }

  public:
    /**
     * @brief Construct a new Sgemm Strided Batched Function object
     */
    SgemmStridedBatchedFunction() { cuda_init(&cublas_handle); }
    /**
     * @brief Construct a new Sgemm Strided Batched Function object
     * @param  function         base class CublasFunction object
     */
    SgemmStridedBatchedFunction(CublasFunction &function) : CublasFunction(function) { cuda_init(&cublas_handle); }
    /**
     * @brief Destroy the Sgemm Strided Batched Function object
     */
    ~SgemmStridedBatchedFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
        cuda_free(&cublas_handle);
    }
};

/**
 * @brief Class of Cgemm3mStridedBatchedFunction
 */
class Cgemm3mStridedBatchedFunction : public CublasFunction {
    cuComplex *Parameter_0_0;
    cuComplex *Parameter_1_0;
    cuComplex *Result_3_0;
    cuComplex *Parameter_0_0_host;
    cuComplex *Parameter_1_0_host;
    std::complex<float> *Result_cpu;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        cgemm3mStridedBatched(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
                              reinterpret_cast<const cuComplex *>(Parameter_0_0),
                              reinterpret_cast<const cuComplex *>(Parameter_1_0),
                              reinterpret_cast<cuComplex *>(Result_3_0), this->batch_count_);
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor(bool random) {
        prepare_tensor_template(&Parameter_0_0, &Parameter_1_0, &Result_3_0, &Parameter_0_0_host, &Parameter_1_0_host,
                                random);
    }
    /**
     * @brief  Function calculation on CPU side
     */
    virtual void matrix_calculation_on_cpu() {
        matrix_calculation_on_cpu_with_data(Parameter_0_0_host, Parameter_1_0_host, Result_3_0, &Result_cpu);
    }
    /**
     * @brief Check the correctness of function calculation result
     */
    virtual int correctness_check() {
        double eps = this->eps == 0.0 ? 1.e-6 : this->eps;
        return check_result(this->batch_count_, Result_3_0, Result_cpu, eps);
    }

  public:
    /**
     * @brief Construct a new Cgemm 3m Strided Batched Function object
     */
    Cgemm3mStridedBatchedFunction() { cuda_init(&cublas_handle); }
    /**
     * @brief Construct a new Cgemm 3m Strided Batched Function object according to base class object
     * @param  function         base class CublasFunction object
     */
    Cgemm3mStridedBatchedFunction(CublasFunction &function) : CublasFunction(function) { cuda_init(&cublas_handle); }
    /**
     * @brief Destroy the Cgemm 3m Strided Batched Function object
     */
    ~Cgemm3mStridedBatchedFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_0_0_host));
        CUDA_SAFE_CALL(cudaFreeHost(Parameter_1_0_host));
        cuda_free(&cublas_handle);
    }
};
