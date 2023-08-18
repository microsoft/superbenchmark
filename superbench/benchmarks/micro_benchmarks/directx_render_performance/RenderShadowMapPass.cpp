// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RenderShadowMapPass.h"

int RenderShadowMapPass::DefineRootParameters(std::vector<CD3DX12_ROOT_PARAMETER> &rootParameters) {
    const int numRootParameters = 1;

    rootParameters.resize(numRootParameters);
    rootParameters[0].InitAsConstantBufferView(0); // obj cb
    return numRootParameters;
}

int RenderShadowMapPass::DefineStaticSamplers(std::vector<CD3DX12_STATIC_SAMPLER_DESC> &samplers) { return 0; }

void RenderShadowMapPass::CreateShaderResourceView(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, int width,
                                                   int height) {
    return;
}

void RenderShadowMapPass::CreateConstantBufferResources(ID3D12Device *device) {
    m_viewCB = std::make_unique<UploadBuffer<ShadowViewConstantBuffer>>(device, 1, true);
}

void RenderShadowMapPass::UpdateConstantBufferData() {
    ShadowViewConstantBuffer viewDBData;
    viewDBData.world = MathHelper::Identity4x4();
    viewDBData.projection = MathHelper::Identity4x4();
    m_viewCB.get()->CopyData(0, viewDBData);
}

void RenderShadowMapPass::BuildPipelineStates(ID3D12Device *device) {
    // Define shader input layout.
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputLayout = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}

    };
    // Create the pipeline state, which includes compiling and loading shaders.
    ComPtr<ID3DBlob> vertexShader = CompileShader(L"Shaders/ShadowMap.hlsl", nullptr, "VS", "vs_5_1");
    ComPtr<ID3DBlob> pixelShader = CompileShader(L"Shaders/ShadowMap.hlsl", nullptr, "PS", "ps_5_1");

    CD3DX12_DEPTH_STENCIL_DESC depthStencilDesc(D3D12_DEFAULT);
    depthStencilDesc.DepthEnable = true;
    depthStencilDesc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    depthStencilDesc.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    depthStencilDesc.StencilEnable = false;

    auto psoDesc = DefinePSODesc(inputLayout, vertexShader, pixelShader);
    psoDesc.DSVFormat = m_deviceResources->m_depthBufferFormat;
    psoDesc.RasterizerState.DepthBias = 100000;
    psoDesc.RasterizerState.DepthBiasClamp = 0.0f;
    psoDesc.RasterizerState.SlopeScaledDepthBias = 1.0f;
    psoDesc.DepthStencilState = depthStencilDesc;

    ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PSOs["ShadowMap"])));
}

void RenderShadowMapPass::SetStatesBeforeDraw(ID3D12GraphicsCommandList *cmdList) {
    // Set necessary state.
    cmdList->SetPipelineState(m_PSOs["ShadowMap"].Get());
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    auto dsv = m_deviceResources->GetDepthStencilView();
    cmdList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);
    auto device = m_deviceResources->GetD3DDevice();
    cmdList->OMSetRenderTargets(1, &GetRenderTargetView(device), true, &dsv);
    // Set root arguments.
    cmdList->SetGraphicsRootConstantBufferView(0, m_viewCB->Resource()->GetGPUVirtualAddress());

    auto ri = m_geometry.get();
    // Set vertex and index buffers
    cmdList->IASetVertexBuffers(0, 1, &ri->VertexBufferView());
    cmdList->IASetIndexBuffer(&ri->IndexBufferView());
    cmdList->IASetPrimitiveTopology(ri->PrimitiveType);
}

void RenderShadowMapPass::Draw(ID3D12GraphicsCommandList *cmdList) { DrawRenderItems(cmdList, m_opts->m_num_object); }
