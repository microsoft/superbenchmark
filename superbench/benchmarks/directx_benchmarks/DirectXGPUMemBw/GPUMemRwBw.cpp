// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <tchar.h>
#include <vector>

#include "GPUMemRwBw.h"

/*
 * @brief Start benchmark.
 */
void GPUMemRwBw::Run() {
    // Create GPU pipeline and device objects.
    LoadPipeline();

    // Prepare data and buffers.
    SIZE_T numElem = this->num_elements_;
    PrepareData(numElem);

    // Load shaders and root signatures.
    LoadAssets();

    m_gpuTimer.init(m_device.Get(), 1);

    // Commit resource allocation command list.
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList *cmdsLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);
    waitForCommandQueue();

    // Start benchmark.
    int loops = opts->num_loop;
    int numWarmUp = opts->num_warm_up;

    double time_ms = MemReadWriteBench(numElem, loops, numWarmUp);
    double bw = numElem * sizeof(Data) * loops / time_ms / 1e6;

    // Output benchmark result.
    cout << "GPUMemBw: " << bw << " GB/s" << endl;
}

/**
 * @brief Allocate resouce on both CPU side and GPU side and construct a array of buffers with given lenght.
 * @param numElement the lenght of data array.

 */
void GPUMemRwBw::PrepareData(SIZE_T numElement) {
    // Prepare CPU side data.
    std::vector<Data> dataA(numElement + 10);
    for (SIZE_T i = 0; i < numElement; i++) {
        dataA[i].v1 = i % 256;
    }
    // Allocate resources on GPU side to take those data.
    UINT64 byteSize = dataA.size() * sizeof(Data);
    if (opts->opt_type == Option::Write || opts->opt_type == Option::ReadWrite) {
        m_inputBuffer =
            CreateDefaultBuffer(m_device.Get(), m_commandList.Get(), dataA.data(), byteSize, m_uploadBuffer);
    }
    // Allocate upload buffer to upload data from CPU to GPU.
    m_device->CreateCommittedResource(
        get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
        get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_outputBuffer));
    // Allocate readback buffer if needed.
    if (opts->check_data && opts->opt_type != Option::Read) {
        // Allocate readback buffer to check result correctness
        m_device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK)),
                                          D3D12_HEAP_FLAG_NONE, get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize)),
                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readbackBuffer));
    }
    // Prepare the parameter buffer of shader.
    UINT8 *pCBDataBegin;
    CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(ParameterBuffer));
    ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&m_constantBuffer)));
    // Fill the constant buffer to pass parameters to GPU.

    ParameterBuffer param;
    // Calculate total number of threads.
    SIZE_T totalThreadNum =
        1LL * (num_dispatch_.x * num_dispatch_.y * num_dispatch_.z) * (num_thread_.x * num_thread_.y * num_thread_.z);
    param.numLoop = numElement / totalThreadNum;
    param.numThread = num_thread_;
    // Upload constant buffer.
    param.numDispatch = num_dispatch_;
    m_constantBuffer->Map(0, nullptr, reinterpret_cast<void **>(&pCBDataBegin));
    memcpy(pCBDataBegin, &param, sizeof(param));
    m_constantBuffer->Unmap(0, nullptr);
}
/**
 * @brief Check result correctness.
 * @param numElement the lenght of data array.
 * @return true if result is correct.
 */
bool GPUMemRwBw::CheckData(SIZE_T numElement) {
    ID3D12CommandAllocator *activeAllocator = m_commandAllocator.Get();
    activeAllocator->Reset();
    m_commandList->Reset(activeAllocator, nullptr);
    // Readback result to check correctness.
    m_commandList->ResourceBarrier(
        1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(m_outputBuffer.Get(), D3D12_RESOURCE_STATE_COMMON,
                                                               D3D12_RESOURCE_STATE_COPY_SOURCE)));
    m_commandList->CopyResource(m_readbackBuffer.Get(), m_outputBuffer.Get());
    m_commandList->ResourceBarrier(
        1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(m_outputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                               D3D12_RESOURCE_STATE_COMMON)));
    m_commandList->Close();
    // Execute copy back and sync.
    ID3D12CommandList *listsToExecute[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(listsToExecute), listsToExecute);
    waitForCommandQueue();
    // Access from CPU.
    Data *mappedData = nullptr;
    m_readbackBuffer->Map(0, nullptr, reinterpret_cast<void **>(&mappedData));
    for (int i = 0; i < numElement; ++i) {
        if ((int)mappedData[i].v1 != i % 256) {
            cout << "FAIL: index " << i << " should be " << i % 256 << " but got " << (int)mappedData[i].v1 << endl;
            break;
        }
    }
    m_readbackBuffer->Unmap(0, nullptr);
    return true;
}
/**
 * @brief Memory read write benchmark.
 * @param numElem the lenght of data array.
 * @return double the time elapsed in ms.
 */
double GPUMemRwBw::MemReadWriteBench(SIZE_T numElem, int loops, int numWarmUp) {
    m_commandAllocator->Reset();
    m_commandList->Reset(m_commandAllocator.Get(), m_PSO.Get());
    // Setup root signature for pipeline.
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    if (opts->opt_type == Option::Write || opts->opt_type == Option::ReadWrite) {
        m_commandList->SetComputeRootShaderResourceView(0, m_inputBuffer->GetGPUVirtualAddress());
    }
    m_commandList->SetComputeRootConstantBufferView(1, m_constantBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(2, m_outputBuffer->GetGPUVirtualAddress());
    // Start test.
    for (int i = 0; i < loops + numWarmUp; i++) {
        if (i == numWarmUp) {
            // Start timestamp.
            m_gpuTimer.start(m_commandList.Get(), 0);
        }
        uint3 dispatch = num_dispatch_;
        m_commandList->Dispatch(dispatch.x, dispatch.y, dispatch.z);
    }
    // Stop timestamp.
    m_gpuTimer.stop(m_commandList.Get(), 0);
    m_gpuTimer.resolveQueryToCPU(m_commandList.Get(), 0);
    // Close, execute (and optionally reset) the command list, and also to use a fence to wait for the command queue.
    m_commandList->Close();
    ID3D12CommandList *listsToExecute[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(listsToExecute), listsToExecute);
    waitForCommandQueue();
    // Get time in ms.
    UINT64 queueFreq;
    m_commandQueue->GetTimestampFrequency(&queueFreq);
    double timestampToMs = (1.0 / queueFreq) * 1000.0;
    D3D12::GPUTimestampPair drawTime = m_gpuTimer.getTimestampPair(0);
    UINT64 dt = drawTime.Stop - drawTime.Start;
    double timeInMs = dt * timestampToMs;
    if (opts->check_data && opts->opt_type != Option::Read) {
        CheckData(numElem);
    }
    return timeInMs;
}
/**
 * @brief Wait until command completed.
 */
void GPUMemRwBw::waitForCommandQueue() {
    // Signal and increment the fence value.
    const UINT64 fenceL = m_fenceValue;
    m_commandQueue->Signal(m_fence, fenceL);
    m_fenceValue++;
    // Wait until command queue is done.
    if (m_fence->GetCompletedValue() < fenceL) {
        m_fence->SetEventOnCompletion(fenceL, m_eventHandle);
        WaitForSingleObject(m_eventHandle, INFINITE);
    }
}
/**
 * @brief Create pipeline including
 *		  create device object, command list, command queue
 *		  and synchronization objects.
 */
void GPUMemRwBw::LoadPipeline() {
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
    cqd3.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    m_device->CreateCommandQueue(&cqd3, IID_PPV_ARGS(&m_commandQueue));
    m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator));
    // Create the command list.
    m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), nullptr,
                                IID_PPV_ARGS(&m_commandList));
    // Command lists are created in the recording state, but there is nothing
    // to record yet. The main loop expects it to be closed, so close it now.
    ThrowIfFailed(m_commandList->Close());
    m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
    m_fenceValue = 1;
    // Create an event handle to use for GPU synchronization.
    m_eventHandle = CreateEvent(0, false, false, 0);
    // Setup GPU resources.
    ID3D12CommandAllocator *activeAllocator = m_commandAllocator.Get();
    activeAllocator->Reset();
    m_commandList->Reset(activeAllocator, nullptr);
}
/*
 * @brief Setup GPU pipeline resource like root signature and shader.
 */
void GPUMemRwBw::LoadAssets() {
    // Prepare root signature, root parameter can be a table, root descriptor or root constants.
    const int nParamter = 3;
    CD3DX12_ROOT_PARAMETER slotRootParameter[nParamter];
    // Perfomance TIP: Order from most frequent to least frequent.
    slotRootParameter[0].InitAsShaderResourceView(0);
    slotRootParameter[1].InitAsConstantBufferView(0);
    slotRootParameter[2].InitAsUnorderedAccessView(0);
    // Create an empty root signature.
    {
        // A root signature is an array of root parameters.
        CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(nParamter, slotRootParameter, 0, nullptr,
                                                D3D12_ROOT_SIGNATURE_FLAG_NONE);
        ComPtr<ID3DBlob> serializedRootSig = nullptr;
        ComPtr<ID3DBlob> errorBlob = nullptr;
        HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                                 serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());
        if (errorBlob != nullptr) {
            ::OutputDebugStringA((char *)errorBlob->GetBufferPointer());
            return;
        }
        ThrowIfFailed(hr);
        ThrowIfFailed(m_device->CreateRootSignature(0, serializedRootSig->GetBufferPointer(),
                                                    serializedRootSig->GetBufferSize(),
                                                    IID_PPV_ARGS(m_rootSignature.GetAddressOf())));
    }
    // Load shader according to user specified.
    switch (opts->opt_type) {
    case Option::Read:
        m_shader = CompileShader(L"ReadWrite.hlsl", nullptr, "Read", "cs_5_0");
        break;
    case Option::Write:
        m_shader = CompileShader(L"ReadWrite.hlsl", nullptr, "Write", "cs_5_0");
        break;
    case Option::ReadWrite:
        m_shader = CompileShader(L"ReadWrite.hlsl", nullptr, "ReadWrite", "cs_5_0");
        break;
    default:
        cout << "Invalid opt type." << endl;
        break;
    }
    // Create the pipeline state, which includes compiling and loading shaders.
    {
        // Describe and create the graphics pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_rootSignature.Get();
        computePsoDesc.CS = {reinterpret_cast<BYTE *>(m_shader->GetBufferPointer()), m_shader->GetBufferSize()};
        computePsoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
        ThrowIfFailed(m_device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_PSO)));
    }
}
/*

 * @brief Create a default buffer and upload data with the upload buffer.
 * @param device the GPU device object.
 * @param cmdList the GPU command list object.
 * @param initData the data that need to upload.
 * @param byteSize the size of data that need to upload.
 * @param uploadBuffer the upload that use for upload data.
 * @return a constant buffer object.
 */
Microsoft::WRL::ComPtr<ID3D12Resource>
GPUMemRwBw::CreateDefaultBuffer(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, const void *initData,
                                UINT64 byteSize, Microsoft::WRL::ComPtr<ID3D12Resource> &uploadBuffer) {
    ComPtr<ID3D12Resource> defaultBuffer;
    // Create target default buffer.
    CD3DX12_HEAP_PROPERTIES DefaultHeap(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC defaultResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
    device->CreateCommittedResource(&DefaultHeap, D3D12_HEAP_FLAG_NONE, &defaultResourceDesc,
                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(defaultBuffer.GetAddressOf()));
    // Create a temporary upload buffer to upload data.
    CD3DX12_HEAP_PROPERTIES UploadHeap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC UploadResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
    device->CreateCommittedResource(&UploadHeap, D3D12_HEAP_FLAG_NONE, &UploadResourceDesc,
                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                    IID_PPV_ARGS(uploadBuffer.GetAddressOf()));
    // Upload data that pass in.
    D3D12_SUBRESOURCE_DATA subResourceData = {};
    subResourceData.pData = initData;
    subResourceData.RowPitch = byteSize;
    subResourceData.SlicePitch = subResourceData.RowPitch;
    // Commit copy command list.
    CD3DX12_RESOURCE_BARRIER WriteBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        defaultBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &WriteBarrier);
    UpdateSubresources<1>(cmdList, defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
    CD3DX12_RESOURCE_BARRIER ReadBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        defaultBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &ReadBarrier);
    return defaultBuffer;
}
