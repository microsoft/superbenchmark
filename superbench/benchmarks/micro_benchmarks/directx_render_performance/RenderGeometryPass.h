// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "RenderApp.h"

class GeometryVertex : Vertex {
  public:
    GeometryVertex() : Vertex() {
        float tx = MathHelper::genRand2N_f(2) - 1;
        float ty = MathHelper::genRand2N_f(2) - 1;
        float tz = MathHelper::genRand2N_f(2) - 1;

        float nx = MathHelper::genRand2N_f(2) - 1;
        float ny = MathHelper::genRand2N_f(2) - 1;
        float nz = MathHelper::genRand2N_f(2) - 1;

        float u = MathHelper::genRand2N_f(1);
        float v = MathHelper::genRand2N_f(1);

        Normal = {tx, ty, tz};
        TangentU = {nx, ny, nz};
        TexC = {u, v};
    }
    GeometryVertex(const DirectX::XMFLOAT3 &p, const DirectX::XMFLOAT3 &n, const DirectX::XMFLOAT2 &uv,
                   const DirectX::XMFLOAT3 &t)
        : Vertex(p.x, p.y, p.z), Normal(n), TangentU(t), TexC(uv) {}
    GeometryVertex(float px, float py, float pz, float nx, float ny, float nz, float tx, float ty, float tz, float u,
                   float v)
        : Vertex(px, py, pz), Normal(nx, ny, nz), TangentU(tx, ty, tz), TexC(u, v) {}
    GeometryVertex(const GeometryVertex &rhs) {
        Normal = rhs.Normal;
        TangentU = rhs.TangentU;
        TexC = rhs.TexC;
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
    }

    DirectX::XMFLOAT3 Normal;
    DirectX::XMFLOAT2 TexC;
    DirectX::XMFLOAT3 TangentU;
};

struct ObjectConstantBuffer {
    DirectX::XMFLOAT4X4 World = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 TexTransform = MathHelper::Identity4x4();
    UINT MaterialIndex;
};

struct BaseViewConstantBuffer {
    DirectX::XMFLOAT4X4 View = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 ViewProj = MathHelper::Identity4x4();
    DirectX::XMFLOAT3 EyePosW = {0.0f, 0.0f, 0.0f};
};

struct MaterialConstantBuffer {
    DirectX::XMFLOAT4 DiffuseAlbedo = {1.0f, 1.0f, 1.0f, 1.0f};
    DirectX::XMFLOAT3 FresnelR0 = {0.01f, 0.01f, 0.01f};
    float Roughness = 0.5f;

    // Used in texture mapping.
    DirectX::XMFLOAT4X4 MatTransform = MathHelper::Identity4x4();

    UINT DiffuseMapIndex = 0;
    UINT NormalMapIndex = 1;
};

class RenderGeometryPass : public RenderApp {
  public:
    RenderGeometryPass(BenchmarkOptions *args) : RenderApp(args) {
        // screen + texture size
        m_numShaderResource = args->m_textureNum + 1;
        m_numPassRenderTargets = 3;
    }
    RenderGeometryPass(BenchmarkOptions *args, HINSTANCE hInstance, HWND hMainWnd, std::wstring &winTitle)
        : RenderApp(args, hInstance, hMainWnd, winTitle) {
        m_numShaderResource = args->m_textureNum + 1;
        m_numPassRenderTargets = 3;
    }
    RenderGeometryPass(const RenderGeometryPass &rhs) = delete;
    RenderGeometryPass &operator=(const RenderGeometryPass &rhs) = delete;
    ~RenderGeometryPass() = default;

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
    virtual void BuildShapeGeometry(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList) override;

    std::unique_ptr<UploadBuffer<ObjectConstantBuffer>> m_objectCB = nullptr;
    std::unique_ptr<UploadBuffer<BaseViewConstantBuffer>> m_viewCB = nullptr;
    std::unique_ptr<UploadBuffer<MaterialConstantBuffer>> m_materialCB = nullptr;
};
