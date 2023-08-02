// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <fstream>
#include <iostream>
#include <tchar.h>
#include <vector>

#include "GPUMemRwBw.h"

/*
 * @brief Start benchmark.
 */
void GPUMemRwBw::Run() {
    // Create GPU pipeline and device objects.
    CreatePipeline();
    // Prepare data and buffers.
    PrepareDataAndBuffer(this->m_num_elements);
    // Load shaders and root signatures.
    LoadAssets();
    // Start benchmark.
    double time_ms = MemReadWriteBench(this->m_num_elements, opts->num_loop, opts->num_warm_up);
    double bw = this->m_num_elements * sizeof(float) * opts->num_loop / time_ms / 1e6;
    // Output benchmark result.
    std::string mode = MemtypeString[static_cast<int>(opts->mem_type)];
    cout << "GPUMemBw: " << mode << " " << opts->size << " " << bw << " GB/s" << endl;
}

/**
 * @brief Allocate resouce on both CPU side and GPU side and construct a array of buffers with given length.
 * @param numElement the length of data array.

 */
void GPUMemRwBw::PrepareDataAndBuffer(SIZE_T numElement) {
    // Prepare CPU side data.
    std::vector<float> dataA(numElement);
    for (SIZE_T i = 0; i < numElement; i++) {
        dataA[i] = i % 256;
    }
    // Allocate resources on GPU side to take those data.
    UINT64 byteSize = dataA.size() * sizeof(float);
    if (opts->mem_type == Memtype::Write || opts->mem_type == Memtype::ReadWrite) {
        m_inputBuffer =
            CreateDefaultBuffer(m_device.Get(), m_commandList.Get(), dataA.data(), byteSize, m_uploadBuffer);
    }
    // Allocate upload buffer to upload data from CPU to GPU.
    ThrowIfFailed(m_device->CreateCommittedResource(
        get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
        get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_outputBuffer)));
    // Allocate readback buffer if needed.
    if (opts->check_data && opts->mem_type != Memtype::Read) {
        // Allocate readback buffer to check result correctness
        ThrowIfFailed(m_device->CreateCommittedResource(
            get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK)), D3D12_HEAP_FLAG_NONE,
            get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize)), D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&m_readbackBuffer)));
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
    SIZE_T totalThreadNum = 1LL * (m_num_dispatch.x * m_num_dispatch.y * m_num_dispatch.z) *
                            (m_num_thread.x * m_num_thread.y * m_num_thread.z);
    param.numLoop = numElement / totalThreadNum;
    param.numThread = m_num_thread;
    // Upload constant buffer.
    param.numDispatch = m_num_dispatch;
    ThrowIfFailed(m_constantBuffer->Map(0, nullptr, reinterpret_cast<void **>(&pCBDataBegin)));
    memcpy(pCBDataBegin, &param, sizeof(param));
    m_constantBuffer->Unmap(0, nullptr);
    // Commit resource allocation command list.
    ExecuteWaitForCommandQueue();
}

/**
 * @brief Check result correctness.
 * @param numElement the length of data array.
 * @return true if result is correct.
 */
bool GPUMemRwBw::CheckData(SIZE_T numElement) {
    // Readback result to check correctness.
    m_commandList->ResourceBarrier(
        1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(m_outputBuffer.Get(), D3D12_RESOURCE_STATE_COMMON,
                                                               D3D12_RESOURCE_STATE_COPY_SOURCE)));
    m_commandList->CopyResource(m_readbackBuffer.Get(), m_outputBuffer.Get());
    m_commandList->ResourceBarrier(
        1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(m_outputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                               D3D12_RESOURCE_STATE_COMMON)));
    // Execute copy back and sync.
    ExecuteWaitForCommandQueue();
    // Access from CPU.
    float *mappedData = nullptr;
    ThrowIfFailed(m_readbackBuffer->Map(0, nullptr, reinterpret_cast<void **>(&mappedData)));
    for (int i = 0; i < numElement; ++i) {
        if ((int)mappedData[i] != i % 256) {
            cout << "Error: check data failed - index " << i << " should be " << i % 256 << " but got "
                 << (int)mappedData[i] << endl;
            break;
        }
    }
    m_readbackBuffer->Unmap(0, nullptr);
    return true;
}

/**
 * @brief Memory read write benchmark.
 * @param numElem the length of data array.
 * @return double the time elapsed in ms.
 */
double GPUMemRwBw::MemReadWriteBench(SIZE_T numElem, int loops, int numWarmUp) {
    // Start test.
    m_gpuTimer.init(m_device.Get(), m_commandQueue.Get(), 1, D3D12::QueueType::compute);
    for (int i = 0; i < loops + numWarmUp; i++) {
        if (i == numWarmUp) {
            // Start timestamp.
            m_gpuTimer.start(m_commandList.Get(), 0);
        }
        UInt3 dispatch = m_num_dispatch;
        m_commandList->Dispatch(dispatch.x, dispatch.y, dispatch.z);
    }
    // Stop timestamp.
    m_gpuTimer.stop(m_commandList.Get(), 0);
    m_gpuTimer.resolveQueryToCPU(m_commandList.Get(), 0);

    // Close, execute (and optionally reset) the command list, and also to use a fence to wait for the command queue.
    ExecuteWaitForCommandQueue();

    // Get time in ms.
    double timeInMs = m_gpuTimer.getElapsedMsByTimestampPair(0);

    if (opts->check_data && opts->mem_type != Memtype::Read) {
        CheckData(numElem);
    }
    return timeInMs;
}

/**
 * @brief Create pipeline including
 *		  create device object, command list, command queue
 *		  and synchronization objects.
 */
void GPUMemRwBw::CreatePipeline() {
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
    cqd3.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(m_device->CreateCommandQueue(&cqd3, IID_PPV_ARGS(&m_commandQueue)));
    ThrowIfFailed(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)));
    // Create the command list.
    ThrowIfFailed(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), nullptr,
                                              IID_PPV_ARGS(&m_commandList)));
    // Create synchronization objects.
    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
    m_fenceValue = 1;
    // Create an event handle to use for GPU synchronization.
    m_eventHandle = CreateEvent(0, false, false, 0);
}

/**
 * @brief Setup GPU pipeline resource including creating root signature, pipeline state and compile shader.
 */
void GPUMemRwBw::LoadAssets() {
    // Prepare root signature, root parameter can be a table, root descriptor or root constants.
    const int nParamter = 3;
    CD3DX12_ROOT_PARAMETER slotRootParameter[nParamter];
    // Bind the SRV, CBV and UAV descriptor tables to the root parameters.
    slotRootParameter[0].InitAsShaderResourceView(0);
    slotRootParameter[1].InitAsConstantBufferView(0);
    slotRootParameter[2].InitAsUnorderedAccessView(0);
    // Create the root signature.
    // A root signature is an array of root parameters.
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(nParamter, slotRootParameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);
    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                             serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());
    if (hr != S_OK || errorBlob != nullptr) {
        std::cout << "Error: " << (char *)errorBlob->GetBufferPointer() << std::endl;
        throw runtime_error("Error: D3D12SerializeRootSignature failed.");
    }
    ThrowIfFailed(m_device->CreateRootSignature(0, serializedRootSig->GetBufferPointer(),
                                                serializedRootSig->GetBufferSize(),
                                                IID_PPV_ARGS(m_rootSignature.GetAddressOf())));
    // Define the number of threads per thread group.
    // LPCSTR pointer obtained from myString.c_str() is only valid as long as the myString object exists.
    std::string x_str = std::to_string(m_num_thread.x);
    LPCSTR x_val = x_str.c_str();
    std::string y_str = std::to_string(m_num_thread.y);
    LPCSTR y_val = y_str.c_str();
    std::string z_str = std::to_string(m_num_thread.z);
    LPCSTR z_val = z_str.c_str();
    D3D_SHADER_MACRO defines[] = {
        {"X", x_val},
        {"Y", y_val},
        {"Z", z_val},
        {nullptr, nullptr} // The last entry must be nullptr to indicate the end of the array
    };
    // Load and Compile shader according to user specified.
    switch (opts->mem_type) {
    case Memtype::Read:
        m_shader = CompileShader(L"ReadWrite.hlsl", defines, "Read", "cs_5_0");
        break;
    case Memtype::Write:
        m_shader = CompileShader(L"ReadWrite.hlsl", defines, "Write", "cs_5_0");
        break;
    case Memtype::ReadWrite:
        m_shader = CompileShader(L"ReadWrite.hlsl", defines, "ReadWrite", "cs_5_0");
        break;
    default:
        std::cout << "Error: Invalid memory type." << std::endl;
        exit(1);
    }
    // Describe and create the graphics pipeline state object (PSO).
    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
    computePsoDesc.pRootSignature = m_rootSignature.Get();
    computePsoDesc.CS = {reinterpret_cast<BYTE *>(m_shader->GetBufferPointer()), m_shader->GetBufferSize()};
    computePsoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    ThrowIfFailed(m_device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_PSO)));

    ExecuteWaitForCommandQueue();

    // Setup root signature for pipeline.
    m_commandList->SetPipelineState(m_PSO.Get());
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());
    if (opts->mem_type == Memtype::Write || opts->mem_type == Memtype::ReadWrite) {
        m_commandList->SetComputeRootShaderResourceView(0, m_inputBuffer->GetGPUVirtualAddress());
    }
    m_commandList->SetComputeRootConstantBufferView(1, m_constantBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(2, m_outputBuffer->GetGPUVirtualAddress());
}

/**
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
    ThrowIfFailed(device->CreateCommittedResource(&DefaultHeap, D3D12_HEAP_FLAG_NONE, &defaultResourceDesc,
                                                  D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  IID_PPV_ARGS(defaultBuffer.GetAddressOf())));
    // Create a temporary upload buffer to upload data.
    CD3DX12_HEAP_PROPERTIES UploadHeap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC UploadResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
    ThrowIfFailed(device->CreateCommittedResource(&UploadHeap, D3D12_HEAP_FLAG_NONE, &UploadResourceDesc,
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                  IID_PPV_ARGS(uploadBuffer.GetAddressOf())));
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

/**
 * @brief Execute the commands and wait until command completed.
 */
void GPUMemRwBw::ExecuteWaitForCommandQueue(DWORD dwMilliseconds) {
    // Close, execute (and optionally reset) the command list, and also to use a fence to wait for the command queue.
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList *listsToExecute[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(listsToExecute), listsToExecute);
    // Signal and increment the fence value.
    const UINT64 fenceL = m_fenceValue;
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fenceL));
    m_fenceValue++;
    // Wait until command queue is done.
    if (m_fence->GetCompletedValue() < fenceL) {
        ThrowIfFailed(m_fence->SetEventOnCompletion(fenceL, m_eventHandle));
        WaitForSingleObject(m_eventHandle, dwMilliseconds);
    }
    // Reset the command allocator and command list.
    ID3D12CommandAllocator *activeAllocator = m_commandAllocator.Get();
    ThrowIfFailed(activeAllocator->Reset());
    ThrowIfFailed(m_commandList->Reset(activeAllocator, nullptr));
}
