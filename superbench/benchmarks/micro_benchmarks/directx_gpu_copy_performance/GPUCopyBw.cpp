// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <tchar.h>
#include <vector>

#include "GPUCopyBw.h"

/**
 * @brief Run the benchmark.
 */
void GPUCopyBw::Run() {
    CreatePipeline();
    double time_ms = CopyResourceBench(opts->size, opts->num_loops, opts->num_warm_up);
    double bw = opts->size * opts->num_loops / time_ms / 1e6;
    string mode = opts->dtoh_enabled ? "dtoh" : "htod";
    cout << mode << ": " << opts->size << "B " << bw << " GB/s" << endl;
}

/**
 * @brief Allocate gpu resources, construct a array of buffers with given size.
 * @param uSize the size of each buffer inside of array.
 */
void GPUCopyBw::InitializeBuffer(SIZE_T uSize) {
    m_defaultBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uSize);

    // The output buffer (created below) is on a default heap, so only the GPU can access it.
    auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(m_device->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &m_defaultBufferDesc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&m_defaultBuffer)));

    // Create upload buffer to upload data to GPU.
    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    ThrowIfFailed(m_device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE, &m_defaultBufferDesc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&m_uploadBuffer)));

    // Create read back buffer if dtoh mode.
    if (opts->dtoh_enabled) {
        auto readbackHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        ThrowIfFailed(m_device->CreateCommittedResource(&readbackHeapProperties, D3D12_HEAP_FLAG_NONE,
                                                        &m_defaultBufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&m_readbackBuffer)));
    }
}

/**
 * @brief Allocate data on CPU side to prepare upload.
 * @param byteSize the size of data to be uploaded.
 */
void GPUCopyBw::PrepareData(SIZE_T byteSize) {
    m_pDataBegin = std::make_unique<uint8_t[]>(byteSize);
    constexpr int uint8_mod = 256;
    for (int j = 0; j < byteSize; j++) {
        m_pDataBegin[j] = static_cast<uint8_t>(j % uint8_mod);
    }
}

/**
 * @brief Check result correctness.
 * @param byteSize the size of data to be checked.
 * @param pData the byte array that expect to be.
 * @return true result is correct.
 */
bool GPUCopyBw::CheckData(SIZE_T byteSize, const uint8_t *pData) {
    if (opts->dtoh_enabled) {
        D3D12_RANGE readbackBufferRange{0, byteSize};
        uint8_t *pReadbackBufferData{};

        // Read back data from GPU.
        ThrowIfFailed(m_readbackBuffer->Map(0, &readbackBufferRange, reinterpret_cast<void **>(&pReadbackBufferData)));
        // Check result correctness.
        for (int i = 0; i < byteSize; i++) {
            if (pData[i] != pReadbackBufferData[i])
                return false;
        }
        D3D12_RANGE emptyRange{0, 0};
        m_readbackBuffer->Unmap(0, &emptyRange);
    }
    return true;
}

/**
 * @brief GPU copy benchmark.
 * @param size the size of data to copy.
 * @param loops the number of copy times to measure the performance.
 * @return double the time elapsed in ms.
 */
double GPUCopyBw::CopyResourceBench(SIZE_T size, int loops, int warm_up) {
    // Prepare CPU side data buffer.
    PrepareData(size);
    // Prepare GPU resources and buffers.
    InitializeBuffer(size);
    // Set data into source buffer.
    PrepareSourceBufferData(m_pDataBegin.get(), size);

    // Run the copy command.
    gpuTimer.init(m_device.Get(), m_commandQueue.Get(), 1, D3D12::QueueType::copy);
    for (int i = 0; i < loops + warm_up; i++) {
        if (i == warm_up) {
            // Start timestamp.
            this->gpuTimer.start(m_commandList.Get(), 0);
        }
        if (opts->htod_enabled) {
            CopyResourceFromUploadToDefault();
        } else if (opts->dtoh_enabled) {
            CopyResourceFromDefaultToReadback();
        }
    }
    // Stop timestamp.
    this->gpuTimer.stop(m_commandList.Get(), 0);
    this->gpuTimer.resolveQueryToCPU(m_commandList.Get(), 0);

    // Close, execute (and optionally reset) the command list, and also to use a fence to wait for the command queue.
    this->ExecuteWaitForCopyQueue();

    // Check if result is correctly copied.
    // The code below assumes that the GPU wrote FLOATs to the buffer.
    if (opts->check_data) {
        bool correctness = CheckData(size, m_pDataBegin.get());
        if (!correctness) {
            std::cout << "Error: Result is not correct!" << std::endl;
        }
    }

    return this->gpuTimer.getElapsedMsByTimestampPair(0);
}

/**
 * @brief Copy data from CPU side to GPU side.
 */
void GPUCopyBw::CopyResourceFromUploadToDefault() {
    m_commandList->CopyResource(m_defaultBuffer.Get(), m_uploadBuffer.Get());
}

/**
 * @brief Copy data from GPU side to GPU side.
 */
void GPUCopyBw::CopyResourceFromDefaultToDefault() {
    m_commandList->CopyResource(m_defaultBuffer.Get(), m_defaultDescBuffer.Get());
}

/**
 * @brief Copy data from GPU side to CPU side.
 */
void GPUCopyBw::CopyResourceFromDefaultToReadback() {
    m_commandList->CopyResource(m_readbackBuffer.Get(), m_defaultBuffer.Get());
}

/**
 * @brief Execute the commands and wait until command completed.
 */
void GPUCopyBw::ExecuteWaitForCopyQueue(DWORD dwMilliseconds) {
    // Close, execute (and optionally reset) the command list, and also to use a fence to wait for the command queue.
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList *listsToExecute[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(listsToExecute), listsToExecute);
    // Signal and increment the fence value.
    const UINT64 fenceL = m_copyFenceValue;
    ThrowIfFailed(m_commandQueue->Signal(m_copyFence.Get(), fenceL));
    m_copyFenceValue++;
    // Wait until command queue is done.
    if (m_copyFence->GetCompletedValue() < fenceL) {
        ThrowIfFailed(m_copyFence->SetEventOnCompletion(fenceL, m_copyEventHandle));
        WaitForSingleObject(m_copyEventHandle, dwMilliseconds);
    }
    // Reset the command allocator and command list.
    ID3D12CommandAllocator *activeAllocator = m_commandAllocator.Get();
    ThrowIfFailed(activeAllocator->Reset());
    ThrowIfFailed(m_commandList->Reset(activeAllocator, nullptr));
}

/**
 * @brief Prepare data of the source buffer of benchmark.
 * @param pData the data that should upload.
 * @param byteSize the size of data.
 */
void GPUCopyBw::PrepareSourceBufferData(const void *pData, SIZE_T byteSize) {
    // Upload data from CPU to upload buffer.
    void *p;
    ThrowIfFailed(m_uploadBuffer->Map(0, nullptr, &p));
    memcpy(p, pData, byteSize);
    m_uploadBuffer->Unmap(0, nullptr);

    if (opts->dtoh_enabled) {
        // Upload data from upload to default buffer.
        CopyResourceFromUploadToDefault();
        D3D12_RESOURCE_BARRIER outputBufferResourceBarrier{CD3DX12_RESOURCE_BARRIER::Transition(
            m_defaultBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)};
        m_commandList->ResourceBarrier(1, &outputBufferResourceBarrier);
        ExecuteWaitForCopyQueue();
    }
}

/**
 * @brief Create pipeline including
 *		  create device object, command list, command queue
 *		  and synchronization objects.
 */
void GPUCopyBw::CreatePipeline() {
    UINT dxgiFactoryFlags = 0;

#if _DEBUG
    // Enable the debug layer (requires the Graphics Tools "optional feature").
    // NOTE: Enabling the debug layer after device creation will invalidate the active device.
    {
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();

            // Enable additional debug layers.
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

    ComPtr<IDXGIAdapter1> hardwareAdapter;
    GetHardwareAdapter(factory.Get(), &hardwareAdapter);

    ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));

    D3D12_COMMAND_QUEUE_DESC cqd3 = {};
    cqd3.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    ThrowIfFailed(m_device->CreateCommandQueue(&cqd3, IID_PPV_ARGS(&m_commandQueue)));

    ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_commandAllocator)));

    // Create the command list.
    ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, m_commandAllocator.Get(), nullptr,
                                              IID_PPV_ARGS(&m_commandList)));

    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_copyFence)));
    m_copyFenceValue = 1;
    // Create an event handle to use for GPU synchronization.
    m_copyEventHandle = CreateEvent(0, false, false, 0);
}
