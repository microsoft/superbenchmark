// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "RenderApp.h"

struct DeferredLightUniformsConstantBuffer {
    XMFLOAT4 DeferredLightUniforms_ShadowMapChannelMask = {0.00, 0.00, 0.00, 0.00};
    XMFLOAT2 DeferredLightUniforms_DistanceFadeMAD = {0.00, 0.00};
    float DeferredLightUniforms_ContactShadowLength = 0.00;
    float DeferredLightUniforms_VolumetricScatteringIntensity = 1.00;
    UINT DeferredLightUniforms_ShadowedBits = 3;
    UINT DeferredLightUniforms_LightingChannelMask = 1;
    float PrePadding_DeferredLightUniforms_40 = 0.00;
    float PrePadding_DeferredLightUniforms_44 = 0.00;
    XMFLOAT3 DeferredLightUniforms_Position = {722.74805, 2515.36084, 94.87169};
    float DeferredLightUniforms_InvRadius = 0.00195;
    XMFLOAT3 DeferredLightUniforms_Color = {8.64818, 6.97867, 4.4531};
    float DeferredLightUniforms_FalloffExponent = 8.00;
    XMFLOAT3 DeferredLightUniforms_Direction = {1.00, 0.00, 0.00};
    float DeferredLightUniforms_SpecularScale = 1.00;
    XMFLOAT3 DeferredLightUniforms_Tangent = {0.00, 0.00, 1.00};
    float DeferredLightUniforms_SourceRadius = 0.00;
    XMFLOAT2 DeferredLightUniforms_SpotAngles = {2.00, 1.00};
    float DeferredLightUniforms_SoftSourceRadius = 0.00;
    float DeferredLightUniforms_SourceLength = 0.00;
    float DeferredLightUniforms_RectLightBarnCosAngle = 2652.84375;
    float DeferredLightUniforms_RectLightBarnLength = 5.89947E-43;
};

struct ViewConstantBuffer {
    XMFLOAT4 View_InvDeviceZToWorldZTransform = {0.00, 0.00, 0.10, -1.00000E-08};
    XMFLOAT4 View_TemporalAAParams = {0.00, 1.00, 0.00, 0.00};
    XMFLOAT4 View_BufferSizeAndInvSize = {1384.00, 676.00, 0.00072, 0.00148};
    XMFLOAT4 View_DiffuseOverrideParameter = {0.00, 0.00, 0.00, 1.00};
    XMFLOAT4 View_SpecularOverrideParameter = {0.00, 0.00, 0.00, 1.00};
    XMFLOAT4X4 View_ClipToView = {0.00, 0.48821, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10,
                                  0.00, 0.00,    1.00, 0.00, 0,    0,    0,    0};
    XMFLOAT4X4 View_ViewToClip = {
        1.00, 0.00, 0.00, 0.00, 0.00, 2.04831, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 10.00, 0.00,
    };
    XMFLOAT4X4 View_ScreenToWorld = {-0.04472, 0.00587,    0.48612,   0.00, -0.98725, 0.12963, -0.09239, 0.00,
                                     -7.70195, 2584.20215, 184.65012, 1.00, 0,        0,       0,        0};
    XMFLOAT3 View_WorldCameraOrigin = {-7.70195, 2584.20215, 184.65012};
    float padding0 = 0;

    XMFLOAT3 View_PreViewTranslation = {7.70195, -2584.20215, -184.65012};
    float padding1 = 0;
    XMFLOAT4 View_ScreenPositionScaleBias = {0.49928, -0.49926, 0.49926, 0.49928};

    XMFLOAT4X4 View_TranslatedWorldToClip = {
        1.00, 0.00, 0.00, 0.00, 0.00, 2.04831, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 10.00, 0.00,
    };
    UINT View_StateFrameIndexMod8View_StateFrameIndexMod8 = 1;
    XMFLOAT3 Padding = {0, 0, 0}; // Add padding to maintain 16-byte alignment
};

struct StencilingConstantBuffer {
    XMFLOAT4 StencilingGeometryPosAndScale = {715.04608, -68.84131, -89.77843, 530.0614};
    XMFLOAT4 StencilingConeParameters = {0.00, 0.00, 0.00, 0.00};
    XMFLOAT4X4 StencilingConeTransform = {0.00, 0.00, -0.005,   0.00,  -1.00, 1.00,  0.50, 1.00,
                                          1.00, 0.00, -1.00143, -1.00, 0.00,  -1.00, 0.00, 0.00};

    XMFLOAT3 StencilingPreViewTranslation = {1.00, 0.00, 0.00};
};

struct ShadowProjectionConstantBuffer {
    XMFLOAT4 LightPositionAndInvRadius = {722.74805, 2515.36084, 94.87169, 0.00195};
    XMFLOAT4 PointLightDepthBiasAndProjParameters = {0.025, 0.00, -0.99805, -1.00};
    XMFLOAT4X4 ShadowViewProjectionMatrices[6] = {{0.00, 0.00, -1.00196, -1.00, 0.00, -1.00, 0.00, 0.00, 1.00, 0.00,
                                                   0.00, 0.00, -94.87168, 2515.3606, -725.16437, -722.74805},
                                                  {0.00, 0.00, 1.00196, 1.00, 0.00, -1.00, 0.00, 0.00, -1.00, 0.00,
                                                   0.00, 0.00, 94.87168, 2515.3606, 723.16046, 722.74805},
                                                  {-1.00, 0.00, 0.00, 0.00, 0.00, 0.00, -1.00196, -1.00, 0.00, 1.00,
                                                   0.00, 0.00, -722.74799, -94.87168, 2519.28125, 2515.36084},
                                                  {-1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00196, 1.00, 0.00, -1.00,
                                                   0.00, 0.00, -722.74799, 94.87168, -2521.28516, -2515.36084},
                                                  {-1.00, 0.00, 0.00, 0.00, 0.00, -1.00, 0.00, 0.00, 0.00, 0.00,
                                                   -1.00196, -1.00, -722.74799, 2515.3606, 94.05539, 94.87169},
                                                  {1.00, 0.00, 0.00, 0.00, 0.00, -1.00, 0.00, 0.00, 0.00, 0.00, 1.00196,
                                                   1.00, 722.74799, 2515.3606, -96.05931, -94.87169}};

    float ShadowSharpen = 1;
    float ShadowFadeFraction = 1;
    float InvShadowmapResolution = 0.00098;
};

class RenderLightingPass : public RenderApp {
  public:
    RenderLightingPass(BenchmarkOptions *opts) : RenderApp(opts) {
        m_numShaderResource = 10;
        m_numPassRenderTargets = 1;
    }
    RenderLightingPass(BenchmarkOptions *args, HINSTANCE hInstance, HWND hMainWnd, std::wstring &winTitle)
        : RenderApp(args, hInstance, hMainWnd, winTitle) {}
    RenderLightingPass(const RenderLightingPass &rhs) = delete;
    RenderLightingPass &operator=(const RenderLightingPass &rhs) = delete;
    ~RenderLightingPass() = default;

    void DrawShadowProjection(ID3D12GraphicsCommandList *cmdList);
    void DrawLighting(ID3D12GraphicsCommandList *cmdList);

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

    std::unique_ptr<UploadBuffer<StencilingConstantBuffer>> m_stencilingCB = nullptr;
    std::unique_ptr<UploadBuffer<ViewConstantBuffer>> m_viewCB = nullptr;
    std::unique_ptr<UploadBuffer<DeferredLightUniformsConstantBuffer>> m_lightingCB = nullptr;
    std::unique_ptr<UploadBuffer<ShadowProjectionConstantBuffer>> m_shadowProjectionCB = nullptr;
};
