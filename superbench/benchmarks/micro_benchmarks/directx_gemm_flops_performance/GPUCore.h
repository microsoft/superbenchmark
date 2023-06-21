// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers.
#endif

#include <chrono>
#include <string>
#include <unordered_map>
#include <wrl.h>
#include <DirectXPackedVector.h>
#include <d3d12.h>
#include <d3d12shader.h>
#include <d3dcompiler.h>
#include <directml.h>
#include <dxgi1_6.h>
#include <windowsx.h>

// linker
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "d3dcompiler.lib")

#define DEBUG 0
#define _PRINT_RESULT 0

#if defined(DEBUG)
#include <dxgidebug.h>
#endif

#include "../directx_third_party/DXSampleHelper.h"
#include "../directx_third_party/d3dx12.h"
#include "../directx_utils/D3D12Timer.h"
#include "Options.h"

using namespace std;
using namespace DirectX;
// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

#define SAFE_RELEASE(p)                                                                                                \
    if (p)                                                                                                             \
    (p)->Release()
template <typename T> T *get_rvalue_ptr(T &&v) { return &v; }

class GPUCore {
  public:
    GPUCore(Options *opts) : opts(opts) {}
    ~GPUCore() { FlushCommandQueue(); }

    /**
     * @brief Setup GPU and start benchmark.
     */
    void Run();

    /**
     * @brief Create pipeline including
     *		  create device object, command list, command queue
     *		  and synchronization objects.
     */
    void LoadPipeline();

    /**
     * @brief Prepare input and output data and buffers of the tensor elements..
     */
    template <typename T> void PrepareData(const int m, const int n, const int k);

    /**
     * @brief Setup and compile DirectML operator.
     */
    void LoadAssets(int m, int n, int k, DML_TENSOR_DATA_TYPE dataType);

    /**
     * @brief Initialize and tart the computation job.
     * @return the elapsed time in ms.
     */
    template <typename T> double InitializeExecuteComputeOp(const int m, const int n, const int k);

    /**
     * @brief Wait until command completed.
     */
    void FlushCommandQueue();

    /**
     * @brief Close and execute command list, wait until command completed.
     */
    void CloseExecuteResetWait();

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
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<IDMLDevice> m_dmlDevice;
    ComPtr<IDMLCommandRecorder> m_dmlCommandRecorder;
    ComPtr<IDMLCompiledOperator> m_dmlCompiledOperator;
    ComPtr<IDMLBindingTable> m_bindingTable;

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
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    UINT64 m_currentFence = 0;
    HANDLE m_eventHandle = nullptr;

    // GPU timer.
    D3D12::D3D12Timer gpuTimer;

    // Options.
    Options *opts;
};
