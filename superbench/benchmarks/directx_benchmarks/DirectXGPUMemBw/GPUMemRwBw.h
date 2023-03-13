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

#include "D3D12Timer.h"
#include "Options.h"
#include "d3dx12.h"
#include "helper.h"

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

#define DEBUG 0

struct Data {
    float v1;
};

struct ParameterBuffer {
    int numLoop;
    uint3 numThread;
    uint3 numDispatch;
};

template <typename T> T *get_rvalue_ptr(T &&v) { return &v; }

class GPUMemRwBw {
  public:
    /**
     * @brief Constructor, initialize the options.
     * @param opts, Options for construct.
     * @param usize, the byte size of data array.
     */
    GPUMemRwBw(Options *opts, SIZE_T usize) : opts(opts) {
        // The setting of num_thread_ here according the the shader file.
        num_thread_ = {1, 256, 1};
        num_elements_ = usize / sizeof(float);
        int numThreadGroup = num_elements_ / (num_thread_.x * num_thread_.y * num_thread_.z);
        num_dispatch_ = {1, numThreadGroup, 1};
    }

    /**
     * @brief Destructor, release the fence.
     */
    ~GPUMemRwBw() { CloseHandle(m_fence); }

    /**
     * @brief Start and run the benchmark.
     */
    void Run();

    /**
     * @brief Memory read write benchmark.
     * @param numElem the lenght of data array.
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
    void LoadPipeline();

    /**
     * @brief Setup GPU pipeline resource like root signature and shader.
     */
    void LoadAssets();

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
     * @brief Allocate resouce on both CPU side and GPU side and construct a array of buffers with given lenght.
     * @param numElement the lenght of data array.
     */
    void PrepareData(SIZE_T numElement);

    /**
     * @brief Wait until command completed.
     */
    void waitForCommandQueue();

    /**
     * @brief Check result correctness.
     * @param numElement the lenght of data array.
     * @return true if result is correct.
     */
    bool CheckData(SIZE_T numElement);

  private:
    // Dispatch layout of command.
    uint3 num_dispatch_;
    // Number of elements in data buffer.
    uint64_t num_elements_ = 0;
    // Number of threads each group.
    uint3 num_thread_;

    // Pipeline objects.
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;

    // Upload buffer to upload data from CPU to GPU.
    ComPtr<ID3D12Resource> m_uploadBuffer;

    // Readback buffer to copy data from GPU to CPU for data check.
    ComPtr<ID3D12Resource> m_readbackBuffer;

    // Synchronization objects.
    ID3D12Fence1 *m_fence = nullptr;
    HANDLE m_eventHandle = nullptr;
    UINT64 m_fenceValue = 0;

    // GPU timer.
    D3D12::D3D12Timer m_gpuTimer;

    // User options.
    Options *opts;

    // Output buffer.
    ComPtr<ID3D12Resource> m_outputBuffer = nullptr;

    // Constant buffer.
    ComPtr<ID3D12Resource> m_constantBuffer = nullptr;

    // Input buffer to pass data into GPU.
    ComPtr<ID3D12Resource> m_inputBuffer = nullptr;

    // Root signature of GPU pipeline.
    ComPtr<ID3D12RootSignature> m_rootSignature = nullptr;

    // Pipeline object to execute.
    ComPtr<ID3D12PipelineState> m_PSO;

    // Shader objects that loaded.
    ComPtr<ID3DBlob> m_shader;
};
