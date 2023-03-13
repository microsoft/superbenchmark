// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <tchar.h>
#include <vector>

#include "GPUCopyBw.h"

/**
 * @brief Start benchmark.
 */
void GPUCopyBw::Run() {
    LoadPipeline();
    double time_ms = CopyResourceBench(opts->size, opts->num_loops, opts->num_warm_up);
    double bw = opts->size / time_ms / 1e6;
    cout << "DirectXGPUCopy:" << bw << " GB/s" << endl;
}

/**
 * @brief Allocate gpu resources, construct a array of buffers with given size.
 * @param uSize the size of each buffer inside of array.
 */
void GPUCopyBw::InitializeBuffer(SIZE_T uSize) {
    DefaultVertexBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uSize);

    // The output buffer (created below) is on a default heap, so only the GPU can access it.
    auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(m_device->CreateCommittedResource(&defaultHeapProperties, D3D12_HEAP_FLAG_NONE,
                                                    &DefaultVertexBufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&m_vertexBuffer)));

    // Create upload buffer to upload data to GPU.
    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    ThrowIfFailed(m_device->CreateCommittedResource(&uploadHeapProperties, D3D12_HEAP_FLAG_NONE,
                                                    &DefaultVertexBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                    nullptr, IID_PPV_ARGS(&m_uploadBuffer)));

    // Create read back buffer if is dtoh mode.
    if (opts->dtoh_enabled) {
        auto readbackHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
        D3D12_RESOURCE_DESC readbackBufferDesc{CD3DX12_RESOURCE_DESC::Buffer(uSize)};
        ThrowIfFailed(m_device->CreateCommittedResource(&readbackHeapProperties, D3D12_HEAP_FLAG_NONE,
                                                        &readbackBufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&m_readbackBuffer)));
    }
}

/**
 * @brief Allocate data on CPU side to prepare upload.
 * @param byteSize the size of data to be uploaded.
 * @return pointer on CPU side.
 */
uint8_t *GPUCopyBw::PrepareData(SIZE_T byteSize) {
    uint8_t *pData = new uint8_t[byteSize];
    constexpr int uint8_mod = 256;
    for (int j = 0; j < byteSize; j++) {
        pData[j] = static_cast<uint8_t>(j % uint8_mod);
    }

    m_pDataBegin = pData;
    return pData;
}

/**
 * @brief Copy data from GPU side to CPU side.
 */
void GPUCopyBw::CopyResourceFromDefaultToReadback() {
    m_commandList->CopyResource(m_readbackBuffer.Get(), m_vertexBuffer.Get());
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
 * @brief Memory copy benchmark.
 * @param size the size of each buffer.
 * @param loops the number of copy times to measure the performance.
 * @return double the time elapsed in ms.
 */
double GPUCopyBw::CopyResourceBench(SIZE_T size, int loops, int warm_up) {
    // Prepare CPU side buffer.
    auto pData = PrepareData(size);
    // Prepare GPU resources and buffers.
    InitializeBuffer(size);

    gpuTimer.init(m_device.Get(), 1);

    ID3D12CommandAllocator *activeAllocator = m_commandAllocator.Get();
    activeAllocator->Reset();
    m_commandList->Reset(activeAllocator, nullptr);

    // Set data into source buffer.
    SetDataToBufferMemcpy(pData, size, 0);

    for (int i = 0; i < loops + warm_up; i++) {
        if (i == warm_up) {
            // Start timestamp.
            this->gpuTimer.start(m_commandList.Get(), 0);
        }
        if (opts->htod_enabled) {
            CopyResourceFromUploadToDefault();
        } else if (opts->dtoh_enabled) {
            CopyResourceFromDefaultToReadback();
        } else if (opts->dtod_enabled) {
            CopyResourceFromDefaultToDefault();
        }
    }

    // Stop timestamp.
    this->gpuTimer.stop(m_commandList.Get(), 0);
    this->gpuTimer.resolveQueryToCPU(m_commandList.Get(), 0);

    // Close, execute (and optionally reset) the command list, and also to use a fence to wait for the command queue.
    m_commandList->Close();
    ID3D12CommandList *listsToExecute[] = {m_commandList.Get()};
    this->m_commandQueue->ExecuteCommandLists(ARRAYSIZE(listsToExecute), listsToExecute);
    this->waitForCopyQueue();

    // Get time in ms.
    UINT64 queueFreq;
    this->m_commandQueue->GetTimestampFrequency(&queueFreq);
    double timestampToMs = (1.0 / queueFreq) * 1000.0;
    D3D12::GPUTimestampPair drawTime = gpuTimer.getTimestampPair(0);
    double timeInMs = (drawTime.Stop - drawTime.Start) * timestampToMs;

    // Check if result is correctly copied.
    // The code below assumes that the GPU wrote FLOATs to the buffer.
    if (opts->check_data) {
        CheckData(size, pData);
    }

    return timeInMs;
}

/**
 * @brief Copy data from CPU side to GPU side.
 */
void GPUCopyBw::CopyResourceFromUploadToDefault() {
    m_commandList->CopyResource(m_vertexBuffer.Get(), m_uploadBuffer.Get());
}

/**
 * @brief Copy data from GPU side to GPU side.
 */
void GPUCopyBw::CopyResourceFromDefaultToDefault() {
    m_commandList->CopyResource(m_vertexBuffer.Get(), m_vertexBuffer_dest.Get());
}

/**
 * @brief Wait until command completed.
 */
void GPUCopyBw::waitForCopyQueue() {
    // Signal and increment the fence value.
    const UINT64 fenceL = copyFenceValue;
    m_commandQueue->Signal(copyFence, fenceL);
    copyFenceValue++;

    // Wait until command queue is done.
    if (copyFence->GetCompletedValue() < fenceL) {
        copyFence->SetEventOnCompletion(fenceL, copyEventHandle);
        WaitForSingleObject(copyEventHandle, INFINITE);
    }
}

/**
 * @brief Prepare data of the init state of benchmark
 *        including upload data from CPU side to GPU side.
 * @param pData the data that should upload.
 * @param byteSize the size of data.
 */
void GPUCopyBw::SetDataToBufferMemcpy(const void *pData, SIZE_T byteSize) {
    // Upload data from CPU to GPU.
    void *p;
    m_uploadBuffer->Map(0, nullptr, &p);
    memcpy(p, pData, byteSize);
    m_uploadBuffer->Unmap(0, nullptr);

    if (opts->dtoh_enabled || opts->dtod_enabled) {
        CopyResourceFromUploadToDefault();
        D3D12_RESOURCE_BARRIER outputBufferResourceBarrier{CD3DX12_RESOURCE_BARRIER::Transition(
            m_vertexBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)};
        m_commandList->ResourceBarrier(1, &outputBufferResourceBarrier);
    }
}

/**
 * @brief Create pipeline including
 *		  create device object, command list, command queue
 *		  and synchronization objects.
 */
void GPUCopyBw::LoadPipeline() {
    UINT dxgiFactoryFlags = 0;

#if DEBUG
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
    m_device->CreateCommandQueue(&cqd3, IID_PPV_ARGS(&m_commandQueue));

    m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_commandAllocator));

    // Create the command list.
    m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, m_commandAllocator.Get(), nullptr,
                                IID_PPV_ARGS(&m_commandList));

    // Command lists are created in the recording state, but there is nothing
    // to record yet. The main loop expects it to be closed, so close it now.
    ThrowIfFailed(m_commandList->Close());

    m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copyFence));
    copyFenceValue = 1;
    // Create an event handle to use for GPU synchronization.
    copyEventHandle = CreateEvent(0, false, false, 0);
}
