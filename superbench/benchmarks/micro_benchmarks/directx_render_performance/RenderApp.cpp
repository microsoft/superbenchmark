// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RenderApp.h"
#include "../directx_third_party/d3dx12.h"

RenderApp::RenderApp(BenchmarkOptions *args) {
    if (args == nullptr) {
        throw std::runtime_error("BenchmarkOptions is nullptr");
    }
    m_opts = args;
    m_width = args->m_width;
    m_height = args->m_height;
    m_deviceResources = std::make_unique<DX::DeviceResources>(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_D32_FLOAT,
                                                              m_swapChainBufferCount, D3D_FEATURE_LEVEL_11_0,
                                                              DX::DeviceResources::c_AllowTearing);
}

RenderApp::RenderApp(BenchmarkOptions *args, HINSTANCE hInstance, HWND hMainWnd, std::wstring &winTitle)
    : RenderApp(args) {
    m_hinstance = hInstance;
    m_hMainWnd = hMainWnd;
    m_winTitle = winTitle;
}

RenderApp::~RenderApp() {
    if (m_outfile.is_open()) {
        m_outfile.close();
    }
    if (m_deviceResources) {
        m_deviceResources->WaitForGpu();
    }
}

void RenderApp::Initialize() {
    if (m_deviceResources == nullptr) {
        throw std::runtime_error("DeviceResources is nullptr");
    }
    m_deviceResources->SetWindow(m_hMainWnd, m_width, m_height);
    m_deviceResources->CreateDeviceResources();
    CreateDeviceDependentResources();

    m_deviceResources->CreateWindowSizeDependentResources();
    CreateWindowSizeDependentResources();

    // Wait until initialization is complete.
    // Execute the initialization commands.
    m_deviceResources->WaitForGpu();

    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    m_gpuTimer.init(device, commandQueue, m_maxTimerNum, D3D12::QueueType::compute);
    m_outfile.open(m_opts->m_outfile, std::ios_base::out);
}

void RenderApp::CreateDeviceDependentResources() {
    auto device = m_deviceResources->GetD3DDevice();
    if (device == nullptr) {
        throw std::runtime_error("D3D12Device is nullptr");
    }
    // Create a fence for synchronizing between different frames
    ThrowIfFailed(device->CreateFence(m_deviceResources->GetCurrentFrameIndex(), D3D12_FENCE_FLAG_NONE,
                                      IID_PPV_ARGS(m_fence.ReleaseAndGetAddressOf())));

    // Start off the fence with the current frame index
    uint64_t currentIdx = m_deviceResources->GetCurrentFrameIndex();
    m_deviceResources->GetCommandQueue()->Signal(m_fence.Get(), currentIdx);

    CreateRootSignatures(device);
    BuildPipelineStates(device);
}

void RenderApp::CreateWindowSizeDependentResources() {
    auto device = m_deviceResources->GetD3DDevice();
    auto rtvHeap = m_deviceResources->m_rtvDescriptorHeap.Get();
    auto pCmdList = m_deviceResources->GetCommandList();
    auto cmdListAlloc = m_deviceResources->GetCommandAllocator();
    auto cmdQueue = m_deviceResources->GetCommandQueue();
    if (device == nullptr) {
        throw std::runtime_error("D3D12Device is nullptr");
    }
    if (rtvHeap == nullptr) {
        throw std::runtime_error("RTVDescriptorHeap is nullptr");
    }
    if (pCmdList == nullptr) {
        throw std::runtime_error("CommandList is nullptr");
    }
    if (cmdListAlloc == nullptr) {
        throw std::runtime_error("CommandAllocator is nullptr");
    }
    if (cmdQueue == nullptr) {
        throw std::runtime_error("CommandQueue is nullptr");
    }

    ThrowIfFailed(cmdListAlloc->Reset());
    ThrowIfFailed(pCmdList->Reset(cmdListAlloc, nullptr));

    // Prepare and init GPU resources.
    if (m_numPassRenderTargets > 0)
        m_renderTargets.resize(m_numPassRenderTargets);
    if (m_numShaderResource > 0)
        m_shaderResources.resize(m_numShaderResource);
    CreateRenderTargetView(device, m_width, m_height, rtvHeap);
    CreateShaderResourceView(device, pCmdList, m_width, m_height);

    // Send the command list off to the GPU for processing.
    ThrowIfFailed(pCmdList->Close());
    ID3D12CommandList *commandLists[] = {pCmdList};
    cmdQueue->ExecuteCommandLists(1, commandLists);
}

void RenderApp::CreateRootSignatures(ID3D12Device *device) {
    std::vector<CD3DX12_ROOT_PARAMETER> rootParameters;
    int numRootParameters = DefineRootParameters(rootParameters);
    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
    rootSignatureDesc.NumParameters = numRootParameters;
    rootSignatureDesc.pParameters = rootParameters.data();
    std::vector<CD3DX12_STATIC_SAMPLER_DESC> samplers;
    auto numSamplers = DefineStaticSamplers(samplers);
    rootSignatureDesc.NumStaticSamplers = (UINT)numSamplers;
    rootSignatureDesc.pStaticSamplers = samplers.data();
    rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ID3DBlob *serializedRootSignature = nullptr;
    ID3DBlob *errorBlob = nullptr;
    auto hr = (D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &serializedRootSignature,
                                           &errorBlob));
    if (hr != S_OK || errorBlob != nullptr) {
        std::cout << ((char *)errorBlob->GetBufferPointer()) << std::endl;
    }

    ThrowIfFailed(device->CreateRootSignature(0, serializedRootSignature->GetBufferPointer(),
                                              serializedRootSignature->GetBufferSize(),
                                              IID_PPV_ARGS(&m_rootSignature)));
}

void RenderApp::CreateRenderTargetResource(ID3D12Device *device, UINT width, UINT height, DXGI_FORMAT format,
                                           D3D12_RESOURCE_FLAGS flags,
                                           Microsoft::WRL::ComPtr<ID3D12Resource> &renderTarget) {
    // Create the render target resources:
    D3D12_CLEAR_VALUE m_clearValue = {}; // Specify a clear value for the render target (optional)
    m_clearValue.Format = format;
    m_clearValue.Color[0] = 0.0f; // Red component
    m_clearValue.Color[1] = 0.0f; // Green component
    m_clearValue.Color[2] = 0.0f; // Blue component
    m_clearValue.Color[3] = 1.0f; // Alpha component

    D3D12_HEAP_PROPERTIES heapProperties = {}; // Specify heap properties for the render target (optional)
    heapProperties.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resourceDesc = {}; // Specify resource properties for the render target
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = width;
    resourceDesc.Height = height;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = format;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = flags;

    // Create the render target resource
    ThrowIfFailed(device->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &resourceDesc,
                                                  D3D12_RESOURCE_STATE_COMMON, &m_clearValue,
                                                  IID_PPV_ARGS(&renderTarget)));
}

CD3DX12_CPU_DESCRIPTOR_HANDLE RenderApp::GetRenderTargetView(ID3D12Device *device) {
    const CD3DX12_CPU_DESCRIPTOR_HANDLE rtvDescriptor(m_rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

    return rtvDescriptor;
}

void RenderApp::CreateRenderTargetView(ID3D12Device *device, UINT width, UINT height, ID3D12DescriptorHeap *rtvHeap) {
    D3D12_DESCRIPTOR_HEAP_DESC rtvDescriptorHeapDesc = {};
    rtvDescriptorHeapDesc.NumDescriptors = m_numPassRenderTargets;
    rtvDescriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(device->CreateDescriptorHeap(&rtvDescriptorHeapDesc,
                                               IID_PPV_ARGS(m_rtvDescriptorHeap.ReleaseAndGetAddressOf())));
    m_rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Define the render target properties
    DXGI_FORMAT format = m_renderTargetFormat;                            // Pixel format of the render target
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET; // Specify the resource flags

    // Create the render target resources.
    for (int i = 0; i < m_numPassRenderTargets; ++i) {
        CreateRenderTargetResource(device, width, height, format, flags, m_renderTargets[i]);
    }

    auto rtvHandle = GetRenderTargetView(device);

    // Create a RTV for each custom render target.
    for (UINT i = 0; i < m_numPassRenderTargets; ++i) {
        // Create the RTV descriptor
        device->CreateRenderTargetView(m_renderTargets[i].Get(), nullptr, rtvHandle);
        // Increment the handle to the next descriptor
        rtvHandle.Offset(1, m_rtvDescriptorSize);
    }
}

D3D12_GRAPHICS_PIPELINE_STATE_DESC RenderApp::DefinePSODesc(const std::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
                                                            ComPtr<ID3DBlob> vertexShader,
                                                            ComPtr<ID3DBlob> pixelShader) {
    // Describe and create the graphics pipeline state object (PSO).
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    psoDesc.InputLayout = {inputLayout.data(), (UINT)inputLayout.size()};
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = {reinterpret_cast<UINT8 *>(vertexShader->GetBufferPointer()), vertexShader->GetBufferSize()};
    psoDesc.PS = {reinterpret_cast<UINT8 *>(pixelShader->GetBufferPointer()), pixelShader->GetBufferSize()};

    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN; // No depth-stencil
    psoDesc.NumRenderTargets = m_numPassRenderTargets;
    for (int i = 0; i < m_numPassRenderTargets; i++) {
        psoDesc.RTVFormats[i] = m_renderTargetFormat;
    }
    psoDesc.SampleDesc.Count = 1;
    psoDesc.SampleDesc.Quality = 0;

    return psoDesc;
}

void RenderApp::LoadAssets() {
    auto device = m_deviceResources->GetD3DDevice();
    auto pCmdList = m_deviceResources->GetCommandList();
    auto cmdListAlloc = m_deviceResources->GetCommandAllocator();
    auto cmdQueue = m_deviceResources->GetCommandQueue();

    ThrowIfFailed(cmdListAlloc->Reset());
    ThrowIfFailed(pCmdList->Reset(cmdListAlloc, nullptr));

    CreateConstantBufferResources(device);
    UpdateConstantBufferData();
    BuildShapeGeometry(device, pCmdList);

    ThrowIfFailed(pCmdList->Close());
    ID3D12CommandList *commandLists[] = {pCmdList};
    cmdQueue->ExecuteCommandLists(1, commandLists);

    this->m_deviceResources->WaitForGpu();
}

void RenderApp::Tick() {
    auto device = m_deviceResources->GetD3DDevice();
    auto pCmdList = m_deviceResources->GetCommandList();
    auto cmdListAlloc = m_deviceResources->GetCommandAllocator();
    auto cmdQueue = m_deviceResources->GetCommandQueue();
    Update();
    Render();
    this->m_deviceResources->WaitForGpu();
    CalculateFrameStats();
}

void RenderApp::Update() {
    // Check to see if the GPU is keeping up
    auto const frameIdx = m_frameIndex;
    auto const numBackBuffers = m_deviceResources->GetBackBufferCount();
    uint64_t completedValue = m_fence->GetCompletedValue();
    if ((frameIdx >
         completedValue) // if frame index is reset to zero it may temporarily be smaller than the last GPU signal
        && (frameIdx - completedValue > numBackBuffers)) {
        // GPU not caught up, wait for at least one available frame
        ThrowIfFailed(m_fence->SetEventOnCompletion(frameIdx - numBackBuffers, m_fenceEvent.Get()));
        WaitForSingleObjectEx(m_fenceEvent.Get(), INFINITE, FALSE);
    }
}

void RenderApp::CalculateFrameStats() {
    auto timeInMs = m_gpuTimer.getElapsedMsByTimestampPair(m_gpuTimerIdx);
    m_frameTimeList.push_back(timeInMs);
    m_gpuTimerIdx++;
    if (m_gpuTimerIdx == m_maxTimerNum) {
        m_gpuTimerIdx = 0;
    }

    m_frameIndex++;
    if (m_frameIndex < m_opts->m_warmup) {
        m_frameTimeList.clear();
    } else {
        cout << m_frameTimeList.back() << endl;
        m_outfile << m_frameTimeList.back() << endl;
    }

    if (m_frameIndex == m_opts->m_warmup + m_opts->m_num_frames) {
        // Calculate the median
        double median = 0;
        std::sort(m_frameTimeList.begin(), m_frameTimeList.end());
        int size = m_frameTimeList.size();
        if (m_frameTimeList.size() % 2 == 0) {
            median = (m_frameTimeList[size / 2 - 1] + m_frameTimeList[size / 2]) / 2;
        } else {
            median = m_frameTimeList[size / 2];
        }
        m_outfile << "Mean: " << median << std::endl;
        std::cout << "Mean: " << median << std::endl;
        PostMessage(m_hMainWnd, WM_CLOSE, 0, 0);
    }
}

void RenderApp::ClearRenderTargetView() {
    auto commandList = m_deviceResources->GetCommandList();
    auto device = m_deviceResources->GetD3DDevice();

    // Clear the views.
    auto rtvDescriptor = GetRenderTargetView(device);
    auto const dsvDescriptor = m_deviceResources->GetDepthStencilView();
    float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    std::vector<CD3DX12_CPU_DESCRIPTOR_HANDLE> rtvHandles(m_numPassRenderTargets);
    for (int i = 0; i < m_numPassRenderTargets; i++) {
        commandList->ClearRenderTargetView(rtvDescriptor, clearColor, 0, nullptr);
        rtvHandles[i] = rtvDescriptor;
        rtvDescriptor.Offset(1, m_rtvDescriptorSize);
    }

    rtvDescriptor = GetRenderTargetView(device);
    commandList->ClearDepthStencilView(dsvDescriptor, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    // Indicate that the back buffer will be used as a render target.
    commandList->OMSetRenderTargets(m_numPassRenderTargets, rtvHandles.data(), FALSE, nullptr);

    // Set the viewport and scissor rect.
    auto const viewport = m_deviceResources->GetScreenViewport();
    auto const scissorRect = m_deviceResources->GetScissorRect();
    commandList->RSSetViewports(1, &viewport);
    commandList->RSSetScissorRects(1, &scissorRect);
}

void RenderApp::PrepareRenderTarget(ID3D12GraphicsCommandList *pCommandList) {
    for (int i = 0; i < m_numPassRenderTargets; i++) {
        // Transition from COMMON to RENDER_TARGET
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = m_renderTargets[i].Get();
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        pCommandList->ResourceBarrier(1, &barrier);
    }
}

void RenderApp::RestoreRenderTarget(ID3D12GraphicsCommandList *pCommandList) {
    for (int i = 0; i < m_numPassRenderTargets; i++) {
        // Indicate that the back buffer will now be used to present.
        pCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[i].Get(),
                                                                               D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                                               D3D12_RESOURCE_STATE_COMMON));
    }
}

void RenderApp::Render() {
    auto device = m_deviceResources->GetD3DDevice();
    auto cmdList = m_deviceResources->GetCommandList();
    auto cmdQueue = m_deviceResources->GetCommandQueue();
    m_deviceResources->Prepare();
    PrepareRenderTarget(cmdList);
    ClearRenderTargetView();
    SetStatesBeforeDraw(cmdList);
    eventStart(cmdList);
    Draw(cmdList);
    eventEnd(cmdList);
    RestoreRenderTarget(cmdList);
    m_deviceResources->Present();
    // GPU will signal an increasing value each frame
    m_deviceResources->GetCommandQueue()->Signal(m_fence.Get(), m_frameIndex);
}

void RenderApp::DrawRenderItems(ID3D12GraphicsCommandList *pCmdList, int drawNum) {
    auto ri = m_geometry.get();
    for (int i = 0; i < drawNum; ++i) {
        pCmdList->DrawIndexedInstanced(ri->IndexCount, 1, ri->StartIndexLocation, ri->BaseVertexLocation, 0);
    }
}

void RenderApp::BuildShapeGeometry(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList) {
    // Create random geometry.
    std::unique_ptr<Geometry> geoData = CreateRandomGeometry<Vertex>(m_opts->m_vertexNum, m_opts->m_indexNum);
    m_geometry = std::make_unique<GeometryResource>();
    m_geometry->Create(device, cmdList, geoData);
}