// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers.
#endif

#include <chrono>
#include <shellapi.h>
#include <string>
#include <unordered_map>
#include <wrl.h>

#include <DirectXMath.h>
#include <d3d12.h>
#include <d3d12shader.h>
#include <d3dcompiler.h>
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

#include "D3D12Timer.h"
#include "Options.h"
#include "d3dx12.h"
#include "helper.h"

using namespace std;
using namespace DirectX;
// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

struct Data {
    float v1;
    float v2;
};

struct ParameterBuffer {
    int numLoop;
    int numThread;
};

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
     * @brief Prepare input and output data and buffers.
     * @param NumDataElements the number of elements.
     */
    void PrepareData(int numDataElements);

    /**
     * @brief Setup GPU pipeline resource like root signature and shader.
     */
    void LoadAssets();

    /**
     * @brief Start the computation job.
     * @return the elapsed time in ms.
     */
    void DoComputeWork_MulAdd();

    /**
     * @brief Wait until command completed.
     */
    void FlushCommandQueue();

    /**
     * @brief Helper function for debuging, which print the result array.
     */
    void PrintOutputResult();

    /**
     * @brief Helper function for debuging, which print the source code of shader.
     * @param computeShaderBlob the binary code of shader.
     */
    void PrintAssembleShaderCode(Microsoft::WRL::ComPtr<ID3DBlob> computeShaderBlob);

    /**
     * @brief Create a default buffer and upload data with the upload buffer.
     * @param device the GPU device object.
     * @param cmdList the GPU command list object.
     * @param initData the data that need to upload.
     * @param byteSize the size of data that need to upload.
     * @param UploadBuffer the upload that use for upload data.
     * @return a constant buffer object.
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

    // Input buffer to pass data into GPU.
    ComPtr<ID3D12Resource> m_inputBufferA = nullptr;
    ComPtr<ID3D12Resource> m_inputUploadBufferA = nullptr;
    ComPtr<ID3D12Resource> m_inputBufferB = nullptr;
    ComPtr<ID3D12Resource> m_inputUploadBufferB = nullptr;

    // Output buffer that result output on GPU.
    ComPtr<ID3D12Resource> m_outputBuffer = nullptr;

    // Readback buffer to copy data from GPU side to CPU side.
    ComPtr<ID3D12Resource> m_readBackBuffer = nullptr;

    // Constant buffer of GPU.
    ComPtr<ID3D12Resource> m_constantBuffer = nullptr;

    // Root signature of GPU pipeline.
    ComPtr<ID3D12RootSignature> m_rootSignature = nullptr;

    // Pipeline object to execute.
    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> m_PSOs;

    // Shader objects that loaded.
    std::unordered_map<std::string, ComPtr<ID3DBlob>> m_shaders;

    // Synchronization objects.
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    UINT64 m_currentFence = 0;
    HANDLE m_eventHandle = nullptr;

    // GPU timer.
    D3D12::D3D12Timer gpuTimer;

    // Options.
    Options *opts;
};
