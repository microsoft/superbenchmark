// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers.
#endif

#include <D3Dcompiler.h>
#include <DirectXMath.h>
#include <d3d12.h>
#include <d3d12shader.h>
#include <dxgi1_6.h>
#include <shellapi.h>
#include <string>
#include <wrl.h>

// linker
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")

#if defined(_DEBUG)
#include <dxgidebug.h>
#endif

#include "../directx_third_party/DXSampleHelper.h"
#include "../directx_third_party/d3dx12.h"
#include "../directx_utils/D3D12Timer.h"
#include "BenchmarkOptions.h"

using namespace DirectX;
// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;
using namespace std;

class GPUCopyBw {
  public:
    GPUCopyBw(BenchmarkOptions *opts) : opts(opts) {}
    ~GPUCopyBw() { CloseHandle(m_copyFence.Get()); }

    /**
     * @brief Run the benchmark.
     */
    void Run();

    /**
     * @brief GPU copy benchmark.
     * @param size the size of data to copy.
     * @param loops the number of copy times to measure the performance.
     * @return double the time elapsed in ms.
     */
    double CopyResourceBench(SIZE_T size, int loops, int warm_up);

    /**
     * @brief Create pipeline including
     *		  create device object, command list, command queue
     *		  and synchronization objects.
     */
    void CreatePipeline();

    /**
     * @brief Allocate data on CPU side to prepare upload.
     * @param byteSize the size of data to be uploaded.
     */
    void PrepareData(SIZE_T byteSize);

    /**
     * @brief Allocate gpu resources, construct a array of buffers with given size.
     * @param uSize the size of each buffer inside of array.
     */
    void InitializeBuffer(SIZE_T uSize);

    /**
     * @brief Prepare data of the source buffer of benchmark.
     * @param pData the data that should upload.
     * @param byteSize the size of data.
     */
    void PrepareSourceBufferData(const void *pData, SIZE_T byteSize);

    /**
     * @brief Copy data from CPU side to GPU side.
     */
    void CopyResourceFromUploadToDefault();

    /**
     * @brief Copy data from GPU side to CPU side.
     */
    void CopyResourceFromDefaultToReadback();

    /**
     * @brief Copy data from GPU side to GPU side.
     */
    void CopyResourceFromDefaultToDefault();

    /**
     * @brief Execute the commands and wait until command completed.
     */
    void ExecuteWaitForCopyQueue(DWORD dwMilliseconds = 60000);

    /**
     * @brief Check result correctness.
     * @param byteSize the size of data to be checked.
     * @param pData the byte array that expect to be.
     * @return true result is correct.
     */
    bool CheckData(SIZE_T byteSize, const uint8_t *pData);

  private:
    // Pipeline objects.
    ComPtr<ID3D12Device> m_device = nullptr;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator = nullptr;
    ComPtr<ID3D12CommandQueue> m_commandQueue = nullptr;
    ComPtr<ID3D12GraphicsCommandList> m_commandList = nullptr;

    // App resources.
    // Pointer of CPU size resource.
    std::unique_ptr<uint8_t[]> m_pDataBegin = nullptr;
    // GPU side buffer.
    ComPtr<ID3D12Resource> m_defaultBuffer = nullptr;
    // GPU side buffer as destination if in dtod mode.
    ComPtr<ID3D12Resource> m_defaultDescBuffer = nullptr;
    // Upload buffer to upload data from CPU to GPU.
    ComPtr<ID3D12Resource> m_uploadBuffer = nullptr;
    // Read back buffer to check data correctness.
    ComPtr<ID3D12Resource> m_readbackBuffer = nullptr;
    // Default buffer descriptor.
    D3D12_RESOURCE_DESC m_defaultBufferDesc;

    // Synchronization objects.
    ComPtr<ID3D12Fence1> m_copyFence = nullptr;
    HANDLE m_copyEventHandle = nullptr;
    UINT64 m_copyFenceValue = 0;

    // GPU timer.
    D3D12::D3D12Timer gpuTimer;

    // Options.
    BenchmarkOptions *opts;
};
