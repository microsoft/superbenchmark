// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers.
#endif

#include <DirectXPackedVector.h>
#include <chrono>
#include <d3d12.h>
#include <d3d12shader.h>
#include <d3dcompiler.h>
#include <directml.h>
#include <dxgi1_6.h>
#include <string>
#include <unordered_map>
#include <windowsx.h>
#include <wrl.h>

// linker
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "d3dcompiler.lib")

#if defined(_DEBUG)
#include <dxgidebug.h>
#endif

#include "../directx_third_party/DXSampleHelper.h"
#include "../directx_third_party/d3dx12.h"
#include "../directx_utils/D3D12Timer.h"
#include "BenchmarkOptions.h"

using namespace std;
using namespace DirectX;
// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

template <typename T> T *get_rvalue_ptr(T &&v) { return &v; }

class GPUCore {
  public:
    GPUCore(BenchmarkOptions *opts) : opts(opts) {}
    ~GPUCore() {}

    /**
     * @brief Setup GPU and start benchmark.
     */
    void Run();

    /**
     * @brief Create pipeline including
     *		  create device object, command list, command queue
     *		  and synchronization objects.
     */
    void CreatePipeline();

    /**
     * @brief Prepare input and output data and buffers of the tensor elements..
     */
    template <typename T> void PrepareData(const int m, const int n, const int k);

    /**
     * @brief Create and initialize DML_TENSOR_DESC.
     */
    std::unique_ptr<DML_TENSOR_DESC> CreateTensorDesc(DML_TENSOR_DATA_TYPE dataType, UINT *tensorSizes,
                                                      int dimensionCount);

    /**
     * @brief Setup and compile DirectML operator.
     */
    void SetupAndCompileOp(int m, int n, int k, DML_TENSOR_DATA_TYPE dataType);

    /**
     * @brief Initialize DirectML operator.
     */
    template <typename T> void InitializeOp(int m, int n, int k);

    /**
     * @brief Execute the computation GEMM op.
     * @return the elapsed time in ms.
     */
    double ExecuteComputeOp();

    /**
     * @brief Close and execute command list, wait until command completed.
     */
    void CloseExecuteResetWait(DWORD dwMilliseconds = 300000);

#if defined _PRINT_RESULT
    /**
     * @brief Print the result of the benchmark for debug.
     */
    template <typename T> void PrintResultForDebug(int m, int n);
#endif

    /**
     * @brief Create a default buffer and upload data with the upload buffer.
     * @param device the GPU device object.
     * @param cmdList the GPU command list object.
     * @param initData the data that need to upload.
     * @param byteSize the size of data that need to upload.
     * @param UploadBuffer the upload that use for upload data.
     * @return a default buffer object.
     */
    Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList,
                                                               const void *initData, UINT64 byteSize,
                                                               Microsoft::WRL::ComPtr<ID3D12Resource> &UploadBuffer);

  private:
    // Pipeline objects.
    ComPtr<ID3D12Device> m_device = nullptr;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator = nullptr;
    ComPtr<ID3D12CommandQueue> m_commandQueue = nullptr;
    ComPtr<ID3D12GraphicsCommandList> m_commandList = nullptr;
    ComPtr<IDMLDevice> m_dmlDevice = nullptr;
    ComPtr<IDMLCommandRecorder> m_dmlCommandRecorder = nullptr;
    ComPtr<IDMLCompiledOperator> m_dmlCompiledOperator = nullptr;
    ComPtr<IDMLBindingTable> m_bindingTable = nullptr;
    ComPtr<ID3D12DescriptorHeap> m_descriptorHeap = nullptr;

    // Input buffer to pass data into GPU.
    ComPtr<ID3D12Resource> m_inputBufferA = nullptr;
    ComPtr<ID3D12Resource> m_inputUploadBufferA = nullptr;
    ComPtr<ID3D12Resource> m_inputBufferB = nullptr;
    ComPtr<ID3D12Resource> m_inputUploadBufferB = nullptr;

    // Output buffer that result output on GPU.
    ComPtr<ID3D12Resource> m_outputBuffer = nullptr;

    // Readback buffer to copy data from GPU side to CPU side.
    ComPtr<ID3D12Resource> m_readBackBuffer = nullptr;

    // Synchronization objects.
    ComPtr<ID3D12Fence> m_fence = nullptr;
    UINT64 m_currentFence = 0;
    HANDLE m_eventHandle = nullptr;

    // GPU timer.
    D3D12::D3D12Timer gpuTimer;

    // Options.
    BenchmarkOptions *opts;
};
