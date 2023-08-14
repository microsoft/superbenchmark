// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RenderLightingPass.h"

void RenderLightingPass::CreateConstantBufferResources(ID3D12Device *device) {
    m_stencilingCB = std::make_unique<UploadBuffer<StencilingConstantBuffer>>(device, 1, true);
    m_viewCB = std::make_unique<UploadBuffer<ViewConstantBuffer>>(device, 1, true);
    m_lightingCB = std::make_unique<UploadBuffer<DeferredLightUniformsConstantBuffer>>(device, 1, true);
    m_shadowProjectionCB = std::make_unique<UploadBuffer<ShadowProjectionConstantBuffer>>(device, 1, true);
}

void RenderLightingPass::UpdateConstantBufferData() {
    StencilingConstantBuffer stencilCBData;
    ViewConstantBuffer viewDBData;
    DeferredLightUniformsConstantBuffer lightingCBData;
    ShadowProjectionConstantBuffer shadowProjectionCBData;

    m_stencilingCB.get()->CopyData(0, stencilCBData);
    m_viewCB.get()->CopyData(0, viewDBData);
    m_lightingCB.get()->CopyData(0, lightingCBData);
    m_shadowProjectionCB.get()->CopyData(0, shadowProjectionCBData);
}

void RenderLightingPass::CreateShaderResourceView(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, int width,
                                                  int height) {
    // Create a descriptor heap that will store the SRV:
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = m_numShaderResource;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvDescriptorHeap)));

    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle(m_srvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
    m_cbvSrvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Fill out the heap with actual descriptors.
    for (int i = 0; i < m_numShaderResource; i++) {
        Texture2D(device, cmdList, m_shaderResources[i], width, height, m_colorFormat);
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = m_shaderResources[i]->GetDesc().Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = m_shaderResources[i]->GetDesc().MipLevels;
        device->CreateShaderResourceView(m_shaderResources[i].Get(), &srvDesc, cpuHandle);
        cpuHandle.Offset(m_cbvSrvDescriptorSize);
    }
}

int RenderLightingPass::DefineRootParameters(std::vector<CD3DX12_ROOT_PARAMETER> &rootParameters) {
    const int numRootParameters = 5;
    rootParameters.resize(numRootParameters);
    // Root signature defines what resources are bound to the graphics pipeline.
    int rootParametersIndex = 0;

    // Create root signatures consisting of 3 constant buffers.
    rootParameters[rootParametersIndex].InitAsConstantBufferView(rootParametersIndex, 0,
                                                                 D3D12_SHADER_VISIBILITY_VERTEX);
    rootParametersIndex++;
    rootParameters[rootParametersIndex].InitAsConstantBufferView(rootParametersIndex, 0, D3D12_SHADER_VISIBILITY_ALL);
    rootParametersIndex++;
    rootParameters[rootParametersIndex].InitAsConstantBufferView(rootParametersIndex, 0, D3D12_SHADER_VISIBILITY_PIXEL);
    rootParametersIndex++;
    rootParameters[rootParametersIndex].InitAsConstantBufferView(rootParametersIndex, 0, D3D12_SHADER_VISIBILITY_PIXEL);
    rootParametersIndex++;

    // SRV root parameter
    std::unique_ptr<CD3DX12_DESCRIPTOR_RANGE[]> descriptorRange = std::make_unique<CD3DX12_DESCRIPTOR_RANGE[]>(1);
    descriptorRange[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, m_numShaderResource, 0,
                            0); // Using valid D3D12_DESCRIPTOR_RANGE_TYPE_SRV
    rootParameters[rootParametersIndex].InitAsDescriptorTable(1, descriptorRange.release(),
                                                              D3D12_SHADER_VISIBILITY_PIXEL);
    rootParametersIndex++;

    return numRootParameters;
}

void RenderLightingPass::BuildPipelineStates(ID3D12Device *device) {
    // Define shader input layout.
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputLayout = {D3D12_INPUT_ELEMENT_DESC{
        "POSITION",                                 // SemanticName
        0,                                          // SemanticIndex
        DXGI_FORMAT_R32G32B32_FLOAT,                // Format
        0,                                          // InputSlot
        D3D12_APPEND_ALIGNED_ELEMENT,               // AlignedByteOffset
        D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, // InputSlotClass
        0                                           // InstanceDataStepRate
    }};
    // Create the pipeline state, which includes compiling and loading shaders.
    ComPtr<ID3DBlob> vertexShader =
        CompileShader(L"Shaders/DefferredLightingVertex.hlsl", nullptr, "RadialVertexMain", "vs_5_1");
    ComPtr<ID3DBlob> pixelShader =
        CompileShader(L"Shaders/DefferredLightingPixel.hlsl", nullptr, "DeferredLightPixelMain", "ps_5_1");
    auto psoDesc = DefinePSODesc(inputLayout, vertexShader, pixelShader);
    ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PSOs["deferredLighting"])));

    vertexShader = CompileShader(L"Shaders/DefferredLightingVertex.hlsl", nullptr, "RadialVertexMain", "vs_5_1");
    pixelShader =
        CompileShader(L"Shaders/DefferredLightingPixel.hlsl", nullptr, "MainOnePassPointLightShadowPS", "ps_5_1");
    psoDesc = DefinePSODesc(inputLayout, vertexShader, pixelShader);
    ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PSOs["ShadowProjection"])));
}

void RenderLightingPass::SetStatesBeforeDraw(ID3D12GraphicsCommandList *cmdList) {
    ID3D12DescriptorHeap *ppHeaps[] = {m_srvDescriptorHeap.Get()};
    cmdList->SetDescriptorHeaps(ARRAYSIZE(ppHeaps), ppHeaps);
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    cmdList->SetGraphicsRootConstantBufferView(0, m_stencilingCB.get()->Resource()->GetGPUVirtualAddress());
    cmdList->SetGraphicsRootConstantBufferView(1, m_viewCB.get()->Resource()->GetGPUVirtualAddress());
    cmdList->SetGraphicsRootConstantBufferView(2, m_lightingCB.get()->Resource()->GetGPUVirtualAddress());
    cmdList->SetGraphicsRootConstantBufferView(3, m_shadowProjectionCB.get()->Resource()->GetGPUVirtualAddress());
    cmdList->SetGraphicsRootDescriptorTable(4, m_srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());

    auto ri = m_geometry.get();
    // Set vertex and index buffers
    cmdList->IASetVertexBuffers(0, 1, &ri->VertexBufferView());
    cmdList->IASetIndexBuffer(&ri->IndexBufferView());
    cmdList->IASetPrimitiveTopology(ri->PrimitiveType);
}

void RenderLightingPass::Draw(ID3D12GraphicsCommandList *cmdList) {
    DrawShadowProjection(cmdList);
    DrawLighting(cmdList);
}

void RenderLightingPass::DrawShadowProjection(ID3D12GraphicsCommandList *cmdList) {
    cmdList->SetPipelineState(m_PSOs["ShadowProjection"].Get());
    DrawRenderItems(cmdList, m_opts->m_num_light);
}

void RenderLightingPass::DrawLighting(ID3D12GraphicsCommandList *cmdList) {

    cmdList->SetPipelineState(m_PSOs["deferredLighting"].Get());
    DrawRenderItems(cmdList, m_opts->m_num_light);
}

/*
 * @brief: Get the samplers.
 * @return: The static samplers.
 */
int RenderLightingPass::DefineStaticSamplers(std::vector<CD3DX12_STATIC_SAMPLER_DESC> &samplerData)

{
    int samplersCount = 10;
    samplerData.resize(samplersCount);

    int samplerIndex = 0;
    CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_SceneDepthTextureSampler(
        samplerIndex,                      // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW

    samplerData[samplerIndex++] = SceneTexturesStruct_SceneDepthTextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_GBufferATextureSampler(
        samplerIndex,                      // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
    samplerData[samplerIndex++] = SceneTexturesStruct_GBufferATextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_GBufferBTextureSampler(
        samplerIndex,                      // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
    samplerData[samplerIndex++] = SceneTexturesStruct_GBufferBTextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_GBufferCTextureSampler(
        samplerIndex,                      // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
    samplerData[samplerIndex++] = SceneTexturesStruct_GBufferCTextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_GBufferDTextureSampler(
        samplerIndex,                      // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
    samplerData[samplerIndex++] = SceneTexturesStruct_GBufferDTextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_GBufferETextureSampler(
        samplerIndex,                      // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
    samplerData[samplerIndex++] = SceneTexturesStruct_GBufferETextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_ScreenSpaceAOTextureSampler(
        samplerIndex,                      // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
    samplerData[samplerIndex++] = SceneTexturesStruct_ScreenSpaceAOTextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC LightAttenuationTextureSampler(
        samplerIndex, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP);
    samplerData[samplerIndex++] = LightAttenuationTextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC SceneTexturesStruct_CustomDepthTextureSampler(
        8,                                 // shaderRegister
        D3D12_FILTER_MIN_MAG_MIP_POINT,    // filter
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressU
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,  // addressV
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP); // addressW
    samplerData[samplerIndex++] = SceneTexturesStruct_CustomDepthTextureSampler;
    const CD3DX12_STATIC_SAMPLER_DESC ShadowDepthCubeTextureSampler(9, // shaderRegister
                                                                    D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT, // filter
                                                                    D3D12_TEXTURE_ADDRESS_MODE_CLAMP,      // addressU
                                                                    D3D12_TEXTURE_ADDRESS_MODE_CLAMP,      // addressV
                                                                    D3D12_TEXTURE_ADDRESS_MODE_CLAMP);     // addressW
    samplerData[samplerIndex++] = ShadowDepthCubeTextureSampler;

    return samplersCount;
}
