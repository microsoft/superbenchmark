/**
 * @copyright Copyright (c) Microsoft Corporation
 * @file cublas_function.h
 * @brief Implementation of specific cublas function
 */

#pragma once

#include "cublas_benchmark.h"

/**
 * @brief Class of SgemmFunction
 */
class SgemmFunction : public CublasFunction {
    float *Parameter_0_0; ///< the pointer of the first input data
    float *Parameter_1_0; ///< the pointer of the second input data
    float *Result_3_0;    ///< the pointer of output data
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        sgemm(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
              reinterpret_cast<const float *>(Parameter_0_0), reinterpret_cast<const float *>(Parameter_1_0),
              reinterpret_cast<float *>(Result_3_0));
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor() { CublasFunction::prepare_tensor_float(&Parameter_0_0, &Parameter_1_0, &Result_3_0); }

  public:
    /**
     * @brief Construct a new Sgemm Function object
     */
    SgemmFunction() { this->batch_count_ = 1; }
    /**
     * @brief Construct a new Sgemm Function object
     * @param  function         base class CublasFunction object
     */
    SgemmFunction(CublasFunction &function) : CublasFunction(function) { this->batch_count_ = 1; }
    /**
     * @brief Destroy the Sgemm Function object
     */
    ~SgemmFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
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
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        cgemm(cublas_handle, this->transa_, this->transb_, this->m_, this->n_, this->k_,
              reinterpret_cast<const cuComplex *>(Parameter_0_0), reinterpret_cast<const cuComplex *>(Parameter_1_0),
              reinterpret_cast<cuComplex *>(Result_3_0));
    }
    /**
     * @brief Prepare memory and data of the input and output for kernel running
     */
    virtual void prepare_tensor() {
        CublasFunction::prepare_tensor_cucomplex(&Parameter_0_0, &Parameter_1_0, &Result_3_0);
    }

  public:
    /**
     * @brief Construct a new Cgemm Function object
     */
    CgemmFunction() { this->batch_count_ = 1; }
    /**
     * @brief Construct a new Cgemm Function object
     * @param  function         base class CublasFunction object
     */
    CgemmFunction(CublasFunction &function) : CublasFunction(function) { this->batch_count_ = 1; }
    /**
     * @brief Destroy the Cgemm Function object
     */
    ~CgemmFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        cuda_free(&cublas_handle);
    }
};

/**
 * @brief Class of GemmExFunction
 */
class GemmExFunction : public CublasFunction {
    float *Parameter_0_0;
    float *Parameter_1_0;
    float *Result_3_0;
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
    virtual void prepare_tensor() { CublasFunction::prepare_tensor_float(&Parameter_0_0, &Parameter_1_0, &Result_3_0); }

  public:
    /**
     * @brief Construct a new Gemm Ex Function object
     */
    GemmExFunction() { this->batch_count_ = 1; }
    /**
     * @brief Construct a new Gemm Ex Function object
     * @param  function         base class CublasFunction object
     */
    GemmExFunction(CublasFunction &function) : CublasFunction(function) { this->batch_count_ = 1; }
    /**
     * @brief Destroy the Gemm Ex Function object
     */
    ~GemmExFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        cuda_free(&cublas_handle);
    }
};

/**
 * @brief Class of GemmStridedBatchedExFunction
 */
class GemmStridedBatchedExFunction : public CublasFunction {
    float *Parameter_0_0;
    float *Parameter_1_0;
    float *Result_3_0;
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
    virtual void prepare_tensor() { CublasFunction::prepare_tensor_float(&Parameter_0_0, &Parameter_1_0, &Result_3_0); }

  public:
    /**
     * @brief Construct a new Gemm Strided Batched Ex Function object
     */
    GemmStridedBatchedExFunction() {}
    /**
     * @brief Construct a new Gemm Strided Batched Ex Function object
     * @param  function         base class CublasFunction object
     */
    GemmStridedBatchedExFunction(CublasFunction &function) : CublasFunction(function) {}
    /**
     * @brief Destroy the Gemm Strided Batched Ex Function object
     */
    ~GemmStridedBatchedExFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
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
    virtual void prepare_tensor() { CublasFunction::prepare_tensor_float(&Parameter_0_0, &Parameter_1_0, &Result_3_0); }

  public:
    /**
     * @brief Construct a new Sgemm Strided Batched Function object
     */
    SgemmStridedBatchedFunction() {}
    /**
     * @brief Construct a new Sgemm Strided Batched Function object
     * @param  function         base class CublasFunction object
     */
    SgemmStridedBatchedFunction(CublasFunction &function) : CublasFunction(function) {}
    /**
     * @brief Destroy the Sgemm Strided Batched Function object
     */
    ~SgemmStridedBatchedFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
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
    virtual void prepare_tensor() {
        CublasFunction::prepare_tensor_cucomplex(&Parameter_0_0, &Parameter_1_0, &Result_3_0);
    }

  public:
    /**
     * @brief Construct a new Cgemm 3m Strided Batched Function object
     */
    Cgemm3mStridedBatchedFunction() {}
    /**
     * @brief Construct a new Cgemm 3m Strided Batched Function object according to base class object
     * @param  function         base class CublasFunction object
     */
    Cgemm3mStridedBatchedFunction(CublasFunction &function) : CublasFunction(function) {}
    /**
     * @brief Destroy the Cgemm 3m Strided Batched Function object
     */
    ~Cgemm3mStridedBatchedFunction() {
        // Free contexts
        CUDA_SAFE_CALL(cudaFree(Parameter_0_0));
        CUDA_SAFE_CALL(cudaFree(Parameter_1_0));
        CUDA_SAFE_CALL(cudaFree(Result_3_0));
        cuda_free(&cublas_handle);
    }
};
