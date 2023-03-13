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

#include <chrono>
#include <shellapi.h>
#include <string>
#include <wrl.h>

// linker
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3dcompiler.lib")

#define DEBUG 0

#if defined(DEBUG)
#include <dxgidebug.h>
#endif

#include "D3D12Timer.h"
#include "Options.h"
#include "d3dx12.h"
#include "helper.h"

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
    GPUCopyBw(Options *opts) : opts(opts) {}
    ~GPUCopyBw() {
        CloseHandle(copyFence);
        delete[] m_pDataBegin;
    }

    /**
     * @brief Start benchmark.
     */
    void Run();

    /**
     * @brief Memory copy benchmark.
     * @param size the size of each buffer.
     * @param loops the number of copy times to measure the performance.
     * @param warm_up the number of warm up copy times.
     * @return double the time elapsed in ms.
     */
    double CopyResourceBench(SIZE_T size, int loops, int warm_up);

    /**
     * @brief Create pipeline including
     *		  create device object, command list, command queue
     *		  and synchronization objects.
     */
    void LoadPipeline();

    /**
     * @brief Allocate data on CPU side to prepare upload.
     * @param byteSize the size of data to be uploaded.
     * @return pointer on CPU side.
     */
    uint8_t *PrepareData(SIZE_T byteSize);

    /**
     * @brief Allocate gpu resources, construct a array of buffers with given size.
     * @param uSize the size of each buffer inside of array.
     */
    void InitializeBuffer(SIZE_T uSize);

    /**
     * @brief Prepare data of the init state of benchmark
     *        including upload data from CPU side to GPU side.
     * @param pData the data that should upload.
     * @param byteSize the size of data.
     */
    void SetDataToBufferMemcpy(const void *pData, SIZE_T byteSize);

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
     * @brief Wait until command completed.
     */
    void waitForCopyQueue();

    /**
     * @brief Check result correctness.
     * @param byteSize the size of data to be checked.
     * @param pData the byte array that expect to be.
     * @return true result is correct.
     */
    bool CheckData(SIZE_T byteSize, const uint8_t *pData);

  private:
    // Pipeline objects.
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;

    // App resources.
    // Pointer of CPU size resource.
    UINT8 *m_pDataBegin = nullptr;
    // GPU side buffer.
    ComPtr<ID3D12Resource> m_vertexBuffer;
    // GPU side buffer as destination if in dtod mode.
    ComPtr<ID3D12Resource> m_vertexBuffer_dest;
    // Upload buffer to upload data from CPU to GPU.
    ComPtr<ID3D12Resource> m_uploadBuffer;
    // Read back buffer to check data correctness.
    ComPtr<ID3D12Resource> m_readbackBuffer;
    // Default buffer descriptor.
    D3D12_RESOURCE_DESC DefaultVertexBufferDesc;

    // Synchronization objects.
    ID3D12Fence1 *copyFence = nullptr;
    HANDLE copyEventHandle = nullptr;
    UINT64 copyFenceValue = 0;

    // GPU timer.
    D3D12::D3D12Timer gpuTimer;

    // Options.
    Options *opts;
};
