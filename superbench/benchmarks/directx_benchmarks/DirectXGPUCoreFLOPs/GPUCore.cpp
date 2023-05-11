// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// GPUCore.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <algorithm>
#include <array>
#include <iostream>
#include <tchar.h>
#include <vector>

#include <directml.h>

#include "GPUCore.h"

#define DEBUG 0
#define _PRINT_RESULT 0

/**
 * @brief Setup GPU and start benchmark.
 */
void GPUCore::Run() {
    int m = opts->m;
    int n = opts->n;
    int k = opts->k;

    // Setup GPU objects like device and command list.
    LoadPipeline();

    int loops = opts->num_loops;
    std::cout << "GPUCoreFLOPs" << std::endl;

    switch (opts->mode_precision) {
    case Option::F32: {
        // Prepare input and output data and buffers.
        PrepareData<float>(opts->m, opts->n, opts->k);
        // Setup pipeline and compile operator.
        LoadAssets(opts->m, opts->n, opts->k, DML_TENSOR_DATA_TYPE_FLOAT32);
        for (int i = 0; i < loops; ++i) {
            gpuTimer.init(m_device.Get(), m_commandQueue.Get(), 1, D3D12::QueueType::compute);
            // Do FLOPs job.
            double timeInMs = InitializeExecuteComputeOp<float>(m, n, k);
            auto flops = (int64_t(m) * n * k + m * n) * 2 * 1e-9 / timeInMs;
            std::cout << flops << " TFLOPs" << std::endl;
        }
    } break;
    case Option::F16: {
        PrepareData<uint16_t>(opts->m, opts->n, opts->k);
        LoadAssets(opts->m, opts->n, opts->k, DML_TENSOR_DATA_TYPE_FLOAT16);
        for (int i = 0; i < loops; ++i) {
            gpuTimer.init(m_device.Get(), m_commandQueue.Get(), 1, D3D12::QueueType::compute);
            // Do FLOPs job.
            double timeInMs = InitializeExecuteComputeOp<uint16_t>(m, n, k);
            auto flops = (int64_t(m) * n * k + m * n) * 2 * 1e-9 / timeInMs;
            std::cout << flops << " TFLOPs" << std::endl;
        }
    } break;
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

    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined(_DEBUG)
    // If the project is in a debug build, then enable the Direct3D 12 debug layer.
    // This is optional (starting in DML_FEATURE_LEVEL_5_2) but strongly recommended!

    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

    ThrowIfFailed(DMLCreateDevice(m_device.Get(), dmlCreateDeviceFlags, IID_PPV_ARGS(m_dmlDevice.GetAddressOf())));

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

inline UINT64 DMLCalcBufferTensorSize(DML_TENSOR_DATA_TYPE dataType, UINT tensorElementCount) {
    UINT elementSizeInBytes = 0;
    switch (dataType) {
    case DML_TENSOR_DATA_TYPE_FLOAT32:

        elementSizeInBytes = 4;
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:

        elementSizeInBytes = 2;
        break;

    default:
        return 0; // Invalid data type
    }

    UINT64 minimumImpliedSizeInBytes = 0;

    // Round up to the nearest 4 bytes.
    // Refer to DirectMLX.h
    minimumImpliedSizeInBytes = (tensorElementCount * elementSizeInBytes + 3) & ~3ull;

    return minimumImpliedSizeInBytes;
}

/**
 * @brief Setup and compile DirectML operator.
 */
void GPUCore::LoadAssets(int m, int n, int k, DML_TENSOR_DATA_TYPE dataType) {
    ComPtr<IDMLOperator> dmlOperator;

    // Create DirectML operator(s). Operators represent abstract functions such as "multiply", "reduce",
    // "convolution", or even compound operations such as recurrent neural nets. This example creates an instance of
    // the Identity operator, which applies the function f(x) = x for all elements in a tensor.
    DML_GEMM_OPERATOR_DESC dmlGEMMOperatorDesc{};

    DML_TENSOR_DESC dmlTensorDescA{};
    dmlTensorDescA.Type = DML_TENSOR_TYPE_BUFFER;

    DML_BUFFER_TENSOR_DESC bindingDescA = {};
    UINT tensorSizesA[4] = {1, 1, m, k};
    UINT tensorElementCount = tensorSizesA[0] * tensorSizesA[1] * tensorSizesA[2] * tensorSizesA[3];
    bindingDescA.DataType = dataType;
    bindingDescA.Flags = DML_TENSOR_FLAG_NONE;
    bindingDescA.DimensionCount = ARRAYSIZE(tensorSizesA);
    bindingDescA.Sizes = tensorSizesA;
    bindingDescA.Strides = nullptr;
    bindingDescA.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, tensorElementCount);

    dmlTensorDescA.Desc = &bindingDescA;
    dmlGEMMOperatorDesc.ATensor = &dmlTensorDescA;

    DML_TENSOR_DESC dmlTensorDescB{};
    dmlTensorDescB.Type = DML_TENSOR_TYPE_BUFFER;

    DML_BUFFER_TENSOR_DESC bindingDescB = {};
    UINT tensorSizesB[4] = {1, 1, k, n};
    tensorElementCount = tensorSizesB[0] * tensorSizesB[1] * tensorSizesB[2] * tensorSizesB[3];
    bindingDescB.DataType = dataType;
    bindingDescB.Flags = DML_TENSOR_FLAG_NONE;
    bindingDescB.DimensionCount = ARRAYSIZE(tensorSizesB);
    bindingDescB.Sizes = tensorSizesB;
    bindingDescB.Strides = nullptr;
    bindingDescB.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, tensorElementCount);

    dmlTensorDescB.Desc = &bindingDescB;
    dmlGEMMOperatorDesc.BTensor = &dmlTensorDescB;

    DML_TENSOR_DESC dmlTensorDescC{};
    dmlTensorDescC.Type = DML_TENSOR_TYPE_BUFFER;

    DML_BUFFER_TENSOR_DESC bindingDescC = {};
    UINT tensorSizes[4] = {1, 1, m, n};
    tensorElementCount = tensorSizes[0] * tensorSizes[1] * tensorSizes[2] * tensorSizes[3];
    bindingDescC.DataType = dataType;
    bindingDescC.Flags = DML_TENSOR_FLAG_NONE;
    bindingDescC.DimensionCount = ARRAYSIZE(tensorSizes);
    bindingDescC.Sizes = tensorSizes;
    bindingDescC.Strides = nullptr;
    bindingDescC.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(dataType, tensorElementCount);

    dmlTensorDescC.Desc = &bindingDescC;

    dmlGEMMOperatorDesc.OutputTensor = &dmlTensorDescC;
    dmlGEMMOperatorDesc.CTensor = nullptr;
    dmlGEMMOperatorDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
    dmlGEMMOperatorDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
    dmlGEMMOperatorDesc.Alpha = 1.0f;
    dmlGEMMOperatorDesc.Beta = 0.0f;

    DML_OPERATOR_DESC dmlOperatorDesc{DML_OPERATOR_GEMM, &dmlGEMMOperatorDesc};

    ThrowIfFailed(m_dmlDevice->CreateOperator(&dmlOperatorDesc, IID_PPV_ARGS(dmlOperator.GetAddressOf())));

    ThrowIfFailed(m_dmlDevice->CompileOperator(dmlOperator.Get(), DML_EXECUTION_FLAG_NONE,
                                               IID_PPV_ARGS(m_dmlCompiledOperator.GetAddressOf())));
}

/**
 * @brief Prepare input and output data and buffers of the tensor elements..
 */
template <typename T> void GPUCore::PrepareData(const int m, const int n, const int k) {
    // Define the tensors.
    std::vector<T> dataA(m * k);
    std::vector<T> dataB(n * k);

    // Prepare input data.
    std::fill(dataA.begin(), dataA.end(), 1);
    std::fill(dataB.begin(), dataB.end(), 1);

    UINT64 byteSize = m * k * sizeof(T);

    // Setup input buffer A and upload input data.
    m_inputBufferA =
        CreateDefaultBuffer(m_device.Get(), m_commandList.Get(), dataA.data(), byteSize, m_inputUploadBufferA);

    byteSize = n * k * sizeof(T);
    // Setup input buffer B and upload input data.
    m_inputBufferB =
        CreateDefaultBuffer(m_device.Get(), m_commandList.Get(), dataB.data(), byteSize, m_inputUploadBufferB);

    byteSize = m * n * sizeof(T);
    // Create output buffer.
    m_device->CreateCommittedResource(
        get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
        get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_outputBuffer));

    // Create readback buffer.
    m_device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK)),
                                      D3D12_HEAP_FLAG_NONE, get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(byteSize)),
                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&m_readBackBuffer));
    CloseExecuteResetWait();
}

/**
 * @brief Initialize and tart the computation job.
 * @return the elapsed time in ms.
 */
template <typename T> double GPUCore::InitializeExecuteComputeOp(int m, int n, int k) {
    ComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
    IDMLCompiledOperator *dmlCompiledOperators[] = {m_dmlCompiledOperator.Get()};
    ThrowIfFailed(m_dmlDevice->CreateOperatorInitializer(ARRAYSIZE(dmlCompiledOperators), dmlCompiledOperators,
                                                         IID_PPV_ARGS(dmlOperatorInitializer.GetAddressOf())));

    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = m_dmlCompiledOperator->GetBindingProperties();
    UINT descriptorCount =
        initializeBindingProperties.RequiredDescriptorCount > executeBindingProperties.RequiredDescriptorCount
            ? initializeBindingProperties.RequiredDescriptorCount
            : executeBindingProperties.RequiredDescriptorCount;

    // Create descriptor heaps.
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&descriptorHeapDesc, _uuidof(descriptorHeap),
                                                 (void **)descriptorHeap.GetAddressOf()));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap *d3D12DescriptorHeaps[] = {descriptorHeap.Get()};
    m_commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = descriptorCount;
    ThrowIfFailed(m_dmlDevice->CreateBindingTable(&dmlBindingTableDesc, IID_PPV_ARGS(m_bindingTable.GetAddressOf())));

    // Create the temporary and persistent resources that are necessary for executing an operator.
    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.
    UINT64 temporaryResourceSize =
        max(initializeBindingProperties.TemporaryResourceSize, executeBindingProperties.TemporaryResourceSize);
    UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

    // Bind and initialize the operator on the GPU.
    ComPtr<ID3D12Resource> temporaryBuffer;
    if (temporaryResourceSize != 0) {
        ThrowIfFailed(m_device->CreateCommittedResource(
            get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
            get_rvalue_ptr(
                CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)),
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(temporaryBuffer.GetAddressOf())));

        if (initializeBindingProperties.TemporaryResourceSize != 0) {
            DML_BUFFER_BINDING bufferBinding{temporaryBuffer.Get(), 0, temporaryResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            m_bindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    ComPtr<ID3D12Resource> persistentBuffer;
    if (persistentResourceSize != 0) {
        ThrowIfFailed(m_device->CreateCommittedResource(
            get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
            get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize)), D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(persistentBuffer.GetAddressOf())));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{persistentBuffer.Get(), 0, persistentResourceSize};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        m_bindingTable->BindOutputs(1, &bindingDesc);
    }

    ThrowIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlCommandRecorder)));

    // Record execution of the operator initializer.
    m_dmlCommandRecorder->RecordDispatch(m_commandList.Get(), dmlOperatorInitializer.Get(), m_bindingTable.Get());
    CloseExecuteResetWait();

    // Bind and execute the operator on the GPU.
    m_commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the
    // initializer).
    dmlBindingTableDesc.Dispatchable = m_dmlCompiledOperator.Get();

    ThrowIfFailed(m_bindingTable->Reset(&dmlBindingTableDesc));

    if (temporaryResourceSize != 0) {
        DML_BUFFER_BINDING bufferBinding{temporaryBuffer.Get(), 0, temporaryResourceSize};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        m_bindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceSize != 0) {
        DML_BUFFER_BINDING bufferBinding{persistentBuffer.Get(), 0, persistentResourceSize};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        m_bindingTable->BindPersistentResource(&bindingDesc);
    }

    CloseExecuteResetWait();

    DML_BUFFER_BINDING inputBufferBindingA{m_inputBufferA.Get(), 0, sizeof(T) * m * k};
    DML_BINDING_DESC inputBindingDescA{DML_BINDING_TYPE_BUFFER, &inputBufferBindingA};

    DML_BUFFER_BINDING inputBufferBindingB{m_inputBufferB.Get(), 0, sizeof(T) * n * k};
    DML_BINDING_DESC inputBindingDescB{DML_BINDING_TYPE_BUFFER, &inputBufferBindingB};

    DML_BUFFER_BINDING bufferBinding = {nullptr, 0, 0};
    DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_NONE, &bufferBinding};

    std::array<DML_BINDING_DESC, 3> inputBindings = {inputBindingDescA, inputBindingDescB, inputBindingDesc};
    m_bindingTable->BindInputs(3, inputBindings.data());

    DML_BUFFER_BINDING outputBufferBinding{m_outputBuffer.Get(), 0, sizeof(T) * n * m};
    DML_BINDING_DESC outputBindingDesc{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};

    m_bindingTable->BindOutputs(1, &outputBindingDesc);

    // Execute the compiled GEMM operator and record the GPU time.
    this->gpuTimer.start(m_commandList.Get(), 0);
    m_dmlCommandRecorder->RecordDispatch(m_commandList.Get(), m_dmlCompiledOperator.Get(), m_bindingTable.Get());
    this->gpuTimer.stop(m_commandList.Get(), 0);
    this->gpuTimer.resolveQueryToCPU(m_commandList.Get(), 0);
    CloseExecuteResetWait();

    double timeInMs = this->gpuTimer.getElapsedMsByTimestampPair(0);

    if (_PRINT_RESULT) {
        // The output buffer now contains the result of the identity operator,
        // so read it back if you want the CPU to access it.

        m_commandList->ResourceBarrier(
            1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
                   m_outputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)));

        m_commandList->CopyResource(m_readBackBuffer.Get(), m_outputBuffer.Get());

        CloseExecuteResetWait();
        D3D12_RANGE tensorBufferRange{0, static_cast<SIZE_T>(sizeof(T) * n * m)};
        T *outputBufferData{};
        ThrowIfFailed(m_readBackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void **>(&outputBufferData)));
        std::wstring outputString = L"output tensor: ";
        for (size_t tensorElementIndex{0}; tensorElementIndex < m * n; ++tensorElementIndex, ++outputBufferData) {
            outputString += std::to_wstring(*outputBufferData) + L' ';
        }

        std::wcout << outputString << std::endl;
        OutputDebugStringW(outputString.c_str());
        D3D12_RANGE emptyRange{0, 0};
        m_readBackBuffer->Unmap(0, &emptyRange);
    }

    return timeInMs;
}

/**
 * @brief Close and execute command list, wait until command completed.
 */
void GPUCore::CloseExecuteResetWait() {
    m_commandList->Close();
    ID3D12CommandList *commandLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    FlushCommandQueue();
    ThrowIfFailed(m_commandAllocator->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
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
    CD3DX12_RESOURCE_DESC defaultResourceDesc =
        CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    device->CreateCommittedResource(&DefaultHeap, D3D12_HEAP_FLAG_NONE, &defaultResourceDesc,
                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                    IID_PPV_ARGS(defaultBuffer.GetAddressOf()));

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

    // CD3DX12_RESOURCE_BARRIER WriteBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
    //     defaultBuffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    // cmdList->ResourceBarrier(1, &WriteBarrier);
    UpdateSubresources<1>(cmdList, defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
    CD3DX12_RESOURCE_BARRIER ReadBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        defaultBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &ReadBarrier);

    return defaultBuffer;
}
