// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_stream_utils.hpp"
__device__ constexpr auto kNumLoopUnrollAlias = stream_config::kNumLoopUnroll;

// Function declarations
template <typename T> inline __device__ void Fetch(T &v, const T *p);
template <typename T> inline __device__ void Store(T *p, const T &v);

__global__ void CopyKernel(double *, const double *);
__global__ void ScaleKernel(double *, const double *, const long);
__global__ void AddKernel(double *, const double *, const double *);
__global__ void TriadKernel(double *, const double *, const double *, const long);