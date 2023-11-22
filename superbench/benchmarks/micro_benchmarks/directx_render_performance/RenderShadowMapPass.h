// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "RenderApp.h"

struct ShadowViewConstantBuffer {
    XMFLOAT4X4 world;
    XMFLOAT4X4 projection;
};

class RenderShadowMapPass : public RenderApp {
  public:
    RenderShadowMapPass(BenchmarkOptions *opts) : RenderApp(opts) {}
    RenderShadowMapPass(BenchmarkOptions *args, HINSTANCE hInstance, HWND hMainWnd, std::wstring &winTitle)
        : RenderApp(args, hInstance, hMainWnd, winTitle) {}
    RenderShadowMapPass(const RenderShadowMapPass &rhs) = delete;
    RenderShadowMapPass &operator=(const RenderShadowMapPass &rhs) = delete;
    ~RenderShadowMapPass() = default;

  protected:
    virtual int DefineRootParameters(std::vector<CD3DX12_ROOT_PARAMETER> &rootParameters) override;
    virtual int DefineStaticSamplers(std::vector<CD3DX12_STATIC_SAMPLER_DESC> &samplers) override;
    virtual void CreateShaderResourceView(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, int width,
                                          int height) override;
    virtual void CreateConstantBufferResources(ID3D12Device *device) override;
    virtual void UpdateConstantBufferData() override;
    virtual void BuildPipelineStates(ID3D12Device *device) override;
    virtual void Draw(ID3D12GraphicsCommandList *cmdList) override;
    virtual void SetStatesBeforeDraw(ID3D12GraphicsCommandList *cmdList) override;

    std::unique_ptr<UploadBuffer<ShadowViewConstantBuffer>> m_viewCB = nullptr;
};
