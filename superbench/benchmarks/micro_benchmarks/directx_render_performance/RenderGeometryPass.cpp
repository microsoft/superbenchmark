// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RenderGeometryPass.h"

int RenderGeometryPass::DefineRootParameters(std::vector<CD3DX12_ROOT_PARAMETER> &rootParameters) {
    int numRootParams = 5;
    rootParameters.resize(numRootParams);

    std::unique_ptr<CD3DX12_DESCRIPTOR_RANGE> texTable0 = std::make_unique<CD3DX12_DESCRIPTOR_RANGE>();
    texTable0->Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0);
    std::unique_ptr<CD3DX12_DESCRIPTOR_RANGE> texTable1 = std::make_unique<CD3DX12_DESCRIPTOR_RANGE>();
    texTable1->Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, m_numShaderResource - 1, 1, 0);

    rootParameters[0].InitAsConstantBufferView(0);                                                  // obj cb
    rootParameters[1].InitAsConstantBufferView(1);                                                  // pass cb
    rootParameters[2].InitAsConstantBufferView(2);                                                  // material cb
    rootParameters[3].InitAsDescriptorTable(1, texTable0.release(), D3D12_SHADER_VISIBILITY_PIXEL); // cube texture
    rootParameters[4].InitAsDescriptorTable(1, texTable1.release(), D3D12_SHADER_VISIBILITY_PIXEL); // texture array

    return numRootParams;
}

int RenderGeometryPass::DefineStaticSamplers(std::vector<CD3DX12_STATIC_SAMPLER_DESC> &samplers) {
    int samplersCount = 1;
    samplers.resize(samplersCount);

    CD3DX12_STATIC_SAMPLER_DESC anisotropicWrap(0,                               // shaderRegister
                                                D3D12_FILTER_ANISOTROPIC,        // filter
                                                D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressU
                                                D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressV
                                                D3D12_TEXTURE_ADDRESS_MODE_WRAP, // addressW
                                                0.0f,                            // mipLODBias
                                                8);                              // maxAnisotropy
    samplers[0] = anisotropicWrap;

    return samplersCount;
}

void RenderGeometryPass::CreateShaderResourceView(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, int width,
                                                  int height) {
    // Create a descriptor heap that will store the SRV:
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = m_numShaderResource;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&m_srvDescriptorHeap)));

    CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle(m_srvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
    m_cbvSrvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Whole screen texture.
    TextureCube(device, cmdList, m_shaderResources[0], m_width, m_height, m_colorFormat);
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = m_shaderResources[0]->GetDesc().Format;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    srvDesc.Texture2D.MipLevels = m_shaderResources[0]->GetDesc().MipLevels;
    device->CreateShaderResourceView(m_shaderResources[0].Get(), &srvDesc, cpuHandle);
    cpuHandle.Offset(m_cbvSrvDescriptorSize);

    // Small texture.
    for (int i = 1; i < m_numShaderResource; i++) {
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

void RenderGeometryPass::CreateConstantBufferResources(ID3D12Device *device) {
    m_viewCB = std::make_unique<UploadBuffer<BaseViewConstantBuffer>>(device, 1, true);
    m_objectCB = std::make_unique<UploadBuffer<ObjectConstantBuffer>>(device, 1, true);
    m_materialCB = std::make_unique<UploadBuffer<MaterialConstantBuffer>>(device, 1, true);
}

void RenderGeometryPass::UpdateConstantBufferData() {
    BaseViewConstantBuffer viewCBData;
    ObjectConstantBuffer objectCBData;
    MaterialConstantBuffer materialCBData;
    m_viewCB->CopyData(0, viewCBData);
    m_objectCB->CopyData(0, objectCBData);
    m_materialCB->CopyData(0, materialCBData);
}

void RenderGeometryPass::BuildShapeGeometry(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList) {
    // Create random geometry.
    std::unique_ptr<Geometry> geoData = CreateRandomGeometry<GeometryVertex>(m_opts->m_vertexNum, m_opts->m_indexNum);
    m_geometry = std::make_unique<GeometryResource>();
    m_geometry->Create(device, cmdList, geoData);
}

void RenderGeometryPass::BuildPipelineStates(ID3D12Device *device) {
    std::string textureCount_str = std::to_string(m_numShaderResource - 1);
    LPCSTR textureCount = textureCount_str.c_str();
    D3D_SHADER_MACRO defines[] = {
        {"TEXTURECOUNT", textureCount},
        {nullptr, nullptr}}; // The last entry must be nullptr to indicate the end of the array
    ComPtr<ID3DBlob> vertexShader = CompileShader(L"Shaders/Base.hlsl", defines, "VS", "vs_5_1");
    ComPtr<ID3DBlob> pixelShader = CompileShader(L"Shaders/Base.hlsl", defines, "PS", "ps_5_1");

    // Define shader input layout.
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputLayout = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 32, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    auto psoDesc = DefinePSODesc(inputLayout, vertexShader, pixelShader);
    ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PSOs["deferredBase"])));
}

void RenderGeometryPass::Draw(ID3D12GraphicsCommandList *cmdList) { DrawRenderItems(cmdList, m_opts->m_num_object); }

void RenderGeometryPass::SetStatesBeforeDraw(ID3D12GraphicsCommandList *cmdList) {
    cmdList->SetPipelineState(m_PSOs["deferredBase"].Get());
    cmdList->SetGraphicsRootSignature(m_rootSignature.Get());
    ID3D12DescriptorHeap *heaps[] = {m_srvDescriptorHeap.Get()};
    cmdList->SetDescriptorHeaps(_countof(heaps), heaps);

    cmdList->SetGraphicsRootConstantBufferView(0, m_objectCB.get()->Resource()->GetGPUVirtualAddress());
    cmdList->SetGraphicsRootConstantBufferView(1, m_viewCB.get()->Resource()->GetGPUVirtualAddress());
    cmdList->SetGraphicsRootConstantBufferView(2, m_materialCB.get()->Resource()->GetGPUVirtualAddress());
    CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(m_srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
    cmdList->SetGraphicsRootDescriptorTable(3, srvHandle);
    srvHandle.Offset(1, m_cbvSrvDescriptorSize);
    cmdList->SetGraphicsRootDescriptorTable(4, m_srvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());

    auto ri = m_geometry.get();
    // Set vertex and index buffers
    cmdList->IASetVertexBuffers(0, 1, &ri->VertexBufferView());
    cmdList->IASetIndexBuffer(&ri->IndexBufferView());
    cmdList->IASetPrimitiveTopology(ri->PrimitiveType);
}