// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers.
#endif

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <wrl.h>

#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <d3d12.h>
#include <d3d12shader.h>
#include <dxgi1_6.h>

#include "../directx_third_party/DXSampleHelper.h"
#include "../directx_third_party/d3dx12.h"
#include "../directx_utils/D3D12Timer.h"
#include "BenchmarkOptions.h"

// linker
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")

#if defined(_DEBUG)
#include <dxgidebug.h>
#endif

using namespace DirectX;
// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;
using namespace std;

struct ParameterBuffer {
    int numLoop;
    UInt3 numThread;
    UInt3 numDispatch;
};

template <typename T> T *get_rvalue_ptr(T &&v) { return &v; }

class GPUMemRwBw {
  public:
    /**
     * @brief Constructor, initialize the options.
     * @param opts, Options for construct.
     * @param usize, the byte size of data array.
     */
    GPUMemRwBw(BenchmarkOptions *opts) : opts(opts) {
        // The setting of num_thread need be consistent with the the shader file.
        m_num_thread = opts->num_threads;
        m_num_elements = opts->size / sizeof(float);
        uint32_t numThreadGroup = m_num_elements / (m_num_thread.x * m_num_thread.y * m_num_thread.z);
        m_num_dispatch = {numThreadGroup, 1, 1};
    }

    /**
     * @brief Destructor, release the fence.
     */
    ~GPUMemRwBw() {}

    /**
     * @brief Start and run the benchmark.
     */
    void Run();

    /**
     * @brief Memory read write benchmark.
     * @param numElem the length of data array.
     * @param loops the number of dispatch tiems for measuring the performance.
     * @param numWarmUp the number of warm up dispatch times.
     * @return double the time elapsed in ms.
     */
    double MemReadWriteBench(SIZE_T numElem, int loops, int numWarmUp);

    /**
     * @brief Create pipeline including
     *		  create device object, command list, command queue
     *		  and synchronization objects.
     */
    void CreatePipeline();

    /**
     * @brief Setup GPU pipeline resource including creating root signature, pipeline state and compile shader.
     */
    void LoadAssets();

    /**
     * @brief Allocate resouce on both CPU side and GPU side and construct a array of buffers with given length.
     * @param numElement the length of data array.
     */
    void PrepareDataAndBuffer(SIZE_T numElement);

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
                                                               Microsoft::WRL::ComPtr<ID3D12Resource> &uploadBuffer);

    /**
     * @brief Execute the commands and wait until command completed.
     */
    void ExecuteWaitForCommandQueue(DWORD dwMilliseconds = 30000);

    /**
     * @brief Check result correctness.
     * @param numElement the length of data array.
     * @return true if result is correct.
     */
    bool CheckData(SIZE_T numElement);

  private:
    // Dispatch layout of command.
    UInt3 m_num_dispatch;
    // Number of elements in data buffer.
    uint32_t m_num_elements = 0;
    // Number of threads each group.
    UInt3 m_num_thread;

    // Pipeline objects.
    ComPtr<ID3D12Device> m_device = nullptr;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator = nullptr;
    ComPtr<ID3D12CommandQueue> m_commandQueue = nullptr;
    ComPtr<ID3D12GraphicsCommandList> m_commandList = nullptr;

    // Upload buffer to upload data from CPU to GPU.
    ComPtr<ID3D12Resource> m_uploadBuffer = nullptr;
    // Input buffer to pass data into GPU.
    ComPtr<ID3D12Resource> m_inputBuffer = nullptr;
    // Readback buffer to copy data from GPU to CPU for data check.
    ComPtr<ID3D12Resource> m_readbackBuffer = nullptr;
    // Output buffer.
    ComPtr<ID3D12Resource> m_outputBuffer = nullptr;
    // Constant buffer.
    ComPtr<ID3D12Resource> m_constantBuffer = nullptr;

    // Root signature of GPU pipeline.
    ComPtr<ID3D12RootSignature> m_rootSignature = nullptr;
    // Pipeline object to execute.
    ComPtr<ID3D12PipelineState> m_PSO = nullptr;
    // Shader objects that loaded.
    ComPtr<ID3DBlob> m_shader = nullptr;

    // Synchronization objects.
    ComPtr<ID3D12Fence1> m_fence = nullptr;
    HANDLE m_eventHandle = nullptr;
    UINT64 m_fenceValue = 0;

    // GPU timer.
    D3D12::D3D12Timer m_gpuTimer;

    // User options.
    BenchmarkOptions *opts;
};
