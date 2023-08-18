// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <iostream>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <windows.h>

#include "../directx_third_party/DeviceResources.h"
#include "../directx_utils/D3D12Timer.h"

#include "BenchmarkOptions.h"
#include "GeometryHelper.h"

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace std;

class RenderApp {
  public:
    RenderApp(BenchmarkOptions *args);
    RenderApp(BenchmarkOptions *args, HINSTANCE hInstance, HWND hMainWnd, std::wstring &winTitle);
    RenderApp(const RenderApp &rhs) = delete;
    RenderApp &operator=(const RenderApp &rhs) = delete;
    ~RenderApp();

    /*
     * @brief: Execute the update and render per frame.
     */
    void Tick();
    /*
     * @brief: Initialize the application.
     */
    virtual void Initialize();
    /*
     * @brief: Prepare the data assets needed for render.
     */
    virtual void LoadAssets();
    /*
     * @brief: Calculate the frame stats.
     */
    void CalculateFrameStats();
    /*
     * @brief: Update to run next frame.
     */
    void Update();
    /*
     * @brief: Executes basic render loop .
     */
    void Render();

  protected:
    /*
     * @brief: Define the root parameters.
     * @param: rootParameters The root parameters to be defined.
     * @return: The number of root parameters.
     */
    virtual int DefineRootParameters(std::vector<CD3DX12_ROOT_PARAMETER> &rootParameters) = 0;
    /*
     * @brief: Define the static samplers.
     * @param: samplers The static samplers to be defined.
     * @return: The number of static samplers.
     */
    virtual int DefineStaticSamplers(std::vector<CD3DX12_STATIC_SAMPLER_DESC> &samplers) = 0;
    /*
     * @brief: Build the pipeline states.
     * @param: device The device to build the pipeline states.
     */
    virtual void BuildPipelineStates(ID3D12Device *device) = 0;
    /*
     * @brief: Create the shader resource view.
     * @param: device The device to create the shader resource view.
     * @param: cmdList The command list to create the shader resource view.
     * @param: width The width of the shader resource view.
     * @param: height The height of the shader resource view.
     */
    virtual void CreateShaderResourceView(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, int width,
                                          int height) = 0;
    /*
     * @brief: Create the constant buffer resources.
     * @param: device The device to create the constant buffer resources.
     */
    virtual void CreateConstantBufferResources(ID3D12Device *device) = 0;
    /*
     * @brief: Update the constant buffer data.
     */
    virtual void UpdateConstantBufferData() = 0;
    /*
     * @brief: Render and draw defined by pass.
     * @param: cmdList The command list to draw the render items.
     */
    virtual void Draw(ID3D12GraphicsCommandList *cmdList) = 0;
    /*
     * @brief: Set the states before draw.
     * @param: cmdList The command list to set the states before draw.
     */
    virtual void SetStatesBeforeDraw(ID3D12GraphicsCommandList *cmdList) = 0;
    /*
     * @brief: Create the device dependent resources.
     */
    virtual void CreateDeviceDependentResources();
    /*
     * @brief: Create the window size dependent resources.
     */
    virtual void CreateWindowSizeDependentResources();
    /*
     * @brief: Create the root signature.
     * @param: device The device to create the root signature.
     */
    virtual void CreateRootSignatures(ID3D12Device *device);
    /*
     * @brief: Build the geometry.
     */
    virtual void BuildShapeGeometry(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList);
    /*
     * @brief: Draw the render items.
     * @param: pCmdList The command list to draw the render item.
     * @param: drawTimes The times to draw the render item.
     */
    virtual void DrawRenderItems(ID3D12GraphicsCommandList *pCmdList, int drawTimes);
    /*
     * @brief: Create the render target view.
     * @param: device The device to create the render target view.
     * @param: width The width of the render target view.
     * @param: height The height of the render target view.
     * @param: rtvHeap The descriptor heap to create the render target view.
     */
    virtual void CreateRenderTargetView(ID3D12Device *device, UINT width, UINT height, ID3D12DescriptorHeap *rtvHeap);
    /*
     * @brief: Create the Render target resource.
     * @param: device The device to create the render target resource.
     * @param: width The width of the render target resource.
     * @param: height The height of the render target resource.
     * @param: format The format of the render target resource.
     * @param: flags The flags of the render target resource.
     * @param: renderTarget The render target resource to be created.
     */
    virtual void CreateRenderTargetResource(ID3D12Device *device, UINT width, UINT height, DXGI_FORMAT format,
                                            D3D12_RESOURCE_FLAGS flags,
                                            Microsoft::WRL::ComPtr<ID3D12Resource> &renderTarget);
    /*
     * @brief: Define the pipeline state description.
     * @param: inputLayout The input layout of the pipeline state description.
     * @param: vertexShader The vertex shader of the pipeline state description.
     * @param: pixelShader The pixel shader of the pipeline state description.
     * @return: The pipeline state description.
     */
    D3D12_GRAPHICS_PIPELINE_STATE_DESC DefinePSODesc(const std::vector<D3D12_INPUT_ELEMENT_DESC> &inputLayout,
                                                     ComPtr<ID3DBlob> vertexShader, ComPtr<ID3DBlob> pixelShader);
    /*
     * @brief: Prepare the render target state to draw.
     */
    void PrepareRenderTarget(ID3D12GraphicsCommandList *pCommandList);
    /*
     * @brief: restore render target state.
     */
    void RestoreRenderTarget(ID3D12GraphicsCommandList *pCommandList);
    /*
     * @brief: Clear, bind the render target view and set the viewport and scissor rect.
     */
    void ClearRenderTargetView();
    /*
     * @brief: Get the first render target view of the pass.
     */
    CD3DX12_CPU_DESCRIPTOR_HANDLE GetRenderTargetView(ID3D12Device *device);

    // Window info.
    std::wstring m_winTitle;
    int m_width = 1280;
    int m_height = 720;
    HINSTANCE m_hinstance = nullptr;
    HWND m_hMainWnd = nullptr;
    int m_swapChainBufferCount = 2;
    // Device resources.
    std::unique_ptr<DX::DeviceResources> m_deviceResources;
    D3D_DRIVER_TYPE m_d3dDriverType = D3D_DRIVER_TYPE_HARDWARE;
    // Root signature.
    ComPtr<ID3D12RootSignature> m_rootSignature = nullptr;
    // Render target view.
    ComPtr<ID3D12DescriptorHeap> m_rtvDescriptorHeap = nullptr;
    DXGI_FORMAT m_renderTargetFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
    DXGI_FORMAT m_colorFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    UINT m_numPassRenderTargets = 1;                                     // Number of render targets
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_renderTargets; // Array of render target resources
    UINT m_rtvDescriptorSize = 0;
    // Shader resource view.
    UINT m_numShaderResource = 0; // Number of ShaderResources
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_shaderResources;
    UINT m_cbvSrvDescriptorSize = 0;
    ComPtr<ID3D12DescriptorHeap> m_srvDescriptorHeap = nullptr;
    // PSO objects.
    std::unordered_map<std::string, ComPtr<ID3D12PipelineState>> m_PSOs;
    std::unique_ptr<GeometryResource> m_geometry;
    // A synchronization fence and an event. These members will be used
    // to synchronize the CPU with the GPU so that there will be no
    // contention for the constant buffers.
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    Microsoft::WRL::Wrappers::Event m_fenceEvent;

    // Frame
    UINT64 m_frameIndex = 0;
    vector<double> m_frameTimeList;
    // Benchmark options.
    BenchmarkOptions *m_opts;
    ofstream m_outfile;
    // GPU timer
    D3D12::D3D12Timer m_gpuTimer;
    int m_maxTimerNum = 500;
    int m_gpuTimerIdx = 0;

    void eventStart(ID3D12GraphicsCommandList *pCommandList) { m_gpuTimer.start(pCommandList, m_gpuTimerIdx); }

    void eventEnd(ID3D12GraphicsCommandList *pCommandList) {
        m_gpuTimer.stop(pCommandList, m_gpuTimerIdx);
        m_gpuTimer.resolveQueryToCPU(pCommandList, m_gpuTimerIdx);
    }
};
