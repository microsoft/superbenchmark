// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// GPUCore.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <tchar.h>
#include <vector>

#include "GPUCore.h"

/**
 * @brief Setup GPU and start benchmark.
 */
void GPUCore::Run() {
    // Setup GPU objects like device and command list.
    LoadPipeline();

    // Prepare input and output data and buffers.
    PrepareData(opts->num_data_elements);

    // Setup pipeline and compile shader.
    LoadAssets();

    // Execute setup commands.
    ThrowIfFailed(m_commandList->Close());
    ID3D12CommandList *cmdsLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);
    FlushCommandQueue();

    int loops = opts->num_loops;
    for (int i = 0; i < loops; ++i) {
        gpuTimer.init(m_device.Get(), 1);

        // Do FLOPs job.
        double timeInMs = DoComputeWork_MulAdd();

        // The flops calculation is as following:
        // tflops
        // = the total float operations/the total seconds/1e12
        // = threads_num * flops_in_single_thread / total miliseconds / 1e9
        // = num_data_elements * each loop in the shader conducts 4 flops * num_loops_in_shader / the total miliseconds
        // / 1e9
        auto flops = long(4 * opts->num_data_elements) / timeInMs / 1e9 * opts->num_loops_in_shader;
        std::cout << flops << " TFLOPs" << std::endl;
    }
}

/**
 * @brief Create pipeline including
 *		  create device object, command list, command queue
 *		  and synchronization objects.
 */
void GPUCore::LoadPipeline() {
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

    // Create GPU device object.
    ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)));

    D3D12_COMMAND_QUEUE_DESC queueDesc;
    // Initialize command queue.
    ZeroMemory(&queueDesc, sizeof(queueDesc));

    // Describe and create the command queue.
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    D3D12_COMMAND_QUEUE_DESC cqd3 = {};
    cqd3.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    m_device->CreateCommandQueue(&cqd3, IID_PPV_ARGS(&m_commandQueue));

    m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator));

    m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), nullptr,
                                IID_PPV_ARGS(&m_commandList));

    ThrowIfFailed(m_commandList->Close());
    // Reset the command list to prep for initialization commands.
    ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));

    // Create fence.
    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
    m_currentFence = 1;
    // Create an event handle to use for GPU synchronization.
    m_eventHandle = CreateEvent(0, false, false, 0);
}

/**
 * @brief Prepare input and output data and buffers.
 * @param NumDataElements the number of elements.
 */
void GPUCore::PrepareData(int NumDataElements) {
    std::vector<Data> dataA(NumDataElements);
    std::vector<Data> dataB(NumDataElements);

    // Prepare input data.
    for (float i = 0; i < NumDataElements; i++) {
        dataA[i].v1 = i;
        dataA[i].v2 = -1;

        dataB[i].v1 = 1;
        dataB[i].v2 = i;
    }

    UINT64 byteSize = dataA.size() * sizeof(Data);

    // Setup input buffer A and upload input data.
    m_inputBufferA =
        CreateDefaultBuffer(m_device.Get(), m_commandList.Get(), dataA.data(), byteSize, m_inputUploadBufferA);

    // Setup input buffer B and upload input data.
    m_inputBufferB =
        CreateDefaultBuffer(m_device.Get(), m_commandList.Get(), dataB.data(), byteSize, m_inputUploadBufferB);

    // Create ouput buffer.
    m_device->CreateCommittedResource(
        get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
        get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_outputBuffer));

    // Create readback buffer.
    m_device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK)),
                                      D3D12_HEAP_FLAG_NONE, get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize)),
                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readBackBuffer));

    // Configure the parameters of shader.
    UINT8 *pCBDataBegin;
    CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(1024 * 64);
    ThrowIfFailed(m_device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&m_constantBuffer)));

    // Prepare the constant buffer of shader.
    ParameterBuffer param;
    param.numLoop = opts->num_loops_in_shader;
    param.numThread = opts->thread_num_per_block;
    m_constantBuffer->Map(0, nullptr, reinterpret_cast<void **>(&pCBDataBegin));
    memcpy(pCBDataBegin, &param, sizeof(param));
    m_constantBuffer->Unmap(0, nullptr);
}

/**
 * @brief Setup GPU pipeline resource like root signature and shader.
 */
void GPUCore::LoadAssets() {
    // Root parameter can be a table, root descriptor or root constants.
    CD3DX12_ROOT_PARAMETER slotRootParameter[4];

    // Register the resource view used in shader.
    slotRootParameter[0].InitAsShaderResourceView(0);
    slotRootParameter[1].InitAsShaderResourceView(1);
    slotRootParameter[2].InitAsUnorderedAccessView(0);
    slotRootParameter[3].InitAsConstantBufferView(0);

    // Create an empty root signature.
    {
        // A root signature is an array of root parameters.
        CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(4, slotRootParameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE);

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

    // Compile shader.
    if (opts->mode_precision == Option::F32) {
        m_shaders["MulAddCS"] = CompileShader(L"MulAdd.hlsl", nullptr, "MulAddCS", "cs_5_0");
    } else {
        m_shaders["MulAddCS"] = CompileShader(L"MulAddFloat16.hlsl", nullptr, "MulAddCS", "cs_5_0");
    }

#if DEBUG
    // Print the assemble code of shader.
    PrintAssembleShaderCode(m_shaders["MulAddCS"]);
#endif

    // Create the pipeline state, which includes compiling and loading shaders.
    {
        // Describe and create the graphics pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_rootSignature.Get();
        computePsoDesc.CS = {reinterpret_cast<BYTE *>(m_shaders["MulAddCS"]->GetBufferPointer()),
                             m_shaders["MulAddCS"]->GetBufferSize()};
        computePsoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        ThrowIfFailed(m_device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&m_PSOs["MulAddCS"])));
    }
}

/**
 * @brief Helper function for debuging, which print the source code of shader.
 * @param computeShaderBlob the binary code of shader.
 */
void GPUCore::PrintAssembleShaderCode(Microsoft::WRL::ComPtr<ID3DBlob> computeShaderBlob) {
    ID3DBlob *disassembled_shader = nullptr;
    // Assemble compiled shader code into readable assembly language
    D3DDisassemble(computeShaderBlob->GetBufferPointer(), computeShaderBlob->GetBufferSize(),
                   D3D_DISASM_ENABLE_INSTRUCTION_NUMBERING, nullptr, &disassembled_shader);

    // Convert the assembly result to a string.
    std::string disassembled_shader_string(reinterpret_cast<const char *>(disassembled_shader->GetBufferPointer()),
                                           disassembled_shader->GetBufferSize());

    // Print assembly result.
    std::cout << disassembled_shader_string << std::endl;
}

/**
 * @brief Start the computation job.
 * @return the elapsed time in ms.
 */
double GPUCore::DoComputeWork_MulAdd() {
    m_commandAllocator->Reset();
    m_commandList->Reset(m_commandAllocator.Get(), m_PSOs["MulAddCS"].Get());
    m_commandList->SetComputeRootSignature(m_rootSignature.Get());

    // Setup root signature for input used inside shader.
    m_commandList->SetComputeRootShaderResourceView(0, m_inputBufferA->GetGPUVirtualAddress());
    m_commandList->SetComputeRootShaderResourceView(1, m_inputBufferB->GetGPUVirtualAddress());
    m_commandList->SetComputeRootUnorderedAccessView(2, m_outputBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(3, m_constantBuffer->GetGPUVirtualAddress());

    // Start timestamp.
    this->gpuTimer.start(m_commandList.Get(), 0);

    // Dispatch FLOPs jobs.
    m_commandList->Dispatch(opts->num_data_elements / opts->thread_num_per_block + 1, 1, 1);

    // Stop timestamp.
    this->gpuTimer.stop(m_commandList.Get(), 0);
    this->gpuTimer.resolveQueryToCPU(m_commandList.Get(), 0);

    // Read back result data to check correctness.
    m_commandList->ResourceBarrier(
        1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(m_outputBuffer.Get(), D3D12_RESOURCE_STATE_COMMON,
                                                               D3D12_RESOURCE_STATE_COPY_SOURCE)));
    m_commandList->CopyResource(m_readBackBuffer.Get(), m_outputBuffer.Get());
    m_commandList->ResourceBarrier(
        1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(m_outputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                               D3D12_RESOURCE_STATE_COMMON)));
    m_commandList->Close();

    // Add the command list to the queue for execution.
    ID3D12CommandList *cmdsLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // Wait for the work to finish.
    FlushCommandQueue();

    // Get time in ms.
    UINT64 queueFreq;
    this->m_commandQueue->GetTimestampFrequency(&queueFreq);
    double timestampToMs = (1.0 / queueFreq) * 1000.0;
    D3D12::GPUTimestampPair drawTime = gpuTimer.getTimestampPair(0);
    UINT64 dt = drawTime.Stop - drawTime.Start;
    double timeInMs = dt * timestampToMs;

#if (DEBUG) && (_PRINT_RESULT)
    // Print the caculate result.
    PrintOutputResult();
#endif

    return timeInMs;
}

/**
 * @brief Helper function for debuging, which print the result array.
 */
void GPUCore::PrintOutputResult() {
    // Map the data so we can read it on CPU.
    Data *mappedData = nullptr;
    m_readBackBuffer->Map(0, nullptr, reinterpret_cast<void **>(&mappedData));

    for (int i = 0; i < opts->num_data_elements; ++i) {
        std::cout << "(" << mappedData[i].v1 << ", " << mappedData[i].v2 << ")" << std::endl;
    }

    m_readBackBuffer->Unmap(0, nullptr);
}

/**
 * @brief Wait until command completed.
 */
void GPUCore::FlushCommandQueue() {
    // Signal and increment the fence value.
    const UINT64 fenceL = m_currentFence;
    m_commandQueue->Signal(m_fence.Get(), fenceL);
    m_currentFence++;

    // Wait until command queue is done.
    if (m_fence->GetCompletedValue() < fenceL) {
        m_fence->SetEventOnCompletion(fenceL, m_eventHandle);
        WaitForSingleObject(m_eventHandle, INFINITE);
    }
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
GPUCore::CreateDefaultBuffer(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, const void *initData,
                             UINT64 byteSize, Microsoft::WRL::ComPtr<ID3D12Resource> &uploadBuffer) {
    ComPtr<ID3D12Resource> defaultBuffer;

    // Create the default buffer on GPU side.
    CD3DX12_HEAP_PROPERTIES DefaultHeap(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC defaultResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
    device->CreateCommittedResource(&DefaultHeap, D3D12_HEAP_FLAG_NONE, &defaultResourceDesc,
                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(defaultBuffer.GetAddressOf()));

    // Create upload buffer to upload data.
    CD3DX12_HEAP_PROPERTIES UploadHeap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC UploadResourceDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
    device->CreateCommittedResource(&UploadHeap, D3D12_HEAP_FLAG_NONE, &UploadResourceDesc,
                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                    IID_PPV_ARGS(uploadBuffer.GetAddressOf()));

    // Upload data to GPU side.
    D3D12_SUBRESOURCE_DATA subResourceData = {};
    subResourceData.pData = initData;
    subResourceData.RowPitch = byteSize;
    subResourceData.SlicePitch = subResourceData.RowPitch;

    CD3DX12_RESOURCE_BARRIER WriteBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        defaultBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &WriteBarrier);
    UpdateSubresources<1>(cmdList, defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
    CD3DX12_RESOURCE_BARRIER ReadBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        defaultBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &ReadBarrier);

    return defaultBuffer;
}
