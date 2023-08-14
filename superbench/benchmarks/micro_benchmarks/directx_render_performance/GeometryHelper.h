// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <random>
#include <wrl.h>

#include <DirectXMath.h>

#include "../directx_third_party/DXSampleHelper.h"
#include "../directx_third_party/pch.h"
#include "BufferHelper.h"

using Microsoft::WRL::ComPtr;

namespace MathHelper {
const float Infinity = FLT_MAX;
const float Pi = 3.1415926535f;
// Create identity4*4 matrix
DirectX::XMFLOAT4X4 Identity4x4();
// Returns random float in [0, n).
float genRand2N_f(int n);
// Returns random uint16_t in [0, n).
uint16_t genRand2N_large(int n);
} // namespace MathHelper

// Simple struct to represent a vertex.
class Vertex {
  public:
    Vertex() {
        x = MathHelper::genRand2N_f(2) - 1;
        y = MathHelper::genRand2N_f(2) - 1;
        z = MathHelper::genRand2N_f(2) - 1;
    }
    Vertex(float x, float y, float z) : x(x), y(y), z(z) {}
    Vertex(const Vertex &v) : x(v.x), y(v.y), z(v.z) {}
    float x, y, z; // Position
    // You can add other attributes such as color, normal, texture coordinates etc.
};

// Simple struct to represent a Geometry object.
struct Geometry {
    std::unique_ptr<Vertex[]> vertexData = nullptr;
    std::vector<uint16_t> indexData;
    UINT vertexNum;
    UINT indexNum;
    UINT vertexByteSize;
    UINT indexByteSize;
    UINT vertexByteStride;
};

// Create a random geometry data buffer.
template <class T> std::unique_ptr<Geometry> CreateRandomGeometry(const UINT vertexNum, const UINT indexNum) {
    static_assert(std::is_base_of<Vertex, T>::value, "T must be a Vertex or derived from Vertex");
    std::unique_ptr<Geometry> geo = make_unique<Geometry>();
    // Create the vertices.
    // Allocate memory and reinterpret_cast it to Vertex array
    geo->vertexData.reset(reinterpret_cast<Vertex *>(new T[vertexNum]));

    // Fill in the random vertex data.
    for (UINT i = 0; i < vertexNum; i++) {
        // Here you need to reinterpret_cast it back to T for accessing/modifying
        T &v = reinterpret_cast<T &>(geo->vertexData[i]);
        v = T();
    }

    // Create the indices.
    // Fill in the random index data.
    for (UINT i = 0; i < indexNum; i++) {
        geo->indexData.push_back(MathHelper::genRand2N_large(vertexNum));
    }
    geo->vertexNum = vertexNum;
    geo->indexNum = indexNum;
    geo->vertexByteStride = sizeof(T);
    geo->vertexByteSize = sizeof(T) * vertexNum;
    geo->indexByteSize = sizeof(std::uint16_t) * indexNum;
    return geo;
}

// Helpter class to manage geometry data buffer on GPU.
struct GeometryResource {
    ComPtr<ID3DBlob> VertexBufferCPU = nullptr;
    ComPtr<ID3DBlob> IndexBufferCPU = nullptr;

    ComPtr<ID3D12Resource> VertexBufferGPU = nullptr;
    ComPtr<ID3D12Resource> IndexBufferGPU = nullptr;

    ComPtr<ID3D12Resource> VertexBufferUploader = nullptr;
    ComPtr<ID3D12Resource> IndexBufferUploader = nullptr;

    // Data about the buffers.
    UINT VertexByteStride = 0;
    UINT VertexBufferByteSize = 0;
    DXGI_FORMAT IndexFormat = DXGI_FORMAT_R16_UINT;
    UINT IndexBufferByteSize = 0;

    D3D12_PRIMITIVE_TOPOLOGY PrimitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    UINT IndexCount = 0;
    UINT StartIndexLocation = 0;
    INT BaseVertexLocation = 0;

    /*
     * @brief Get the vertex buffer view.
     */
    D3D12_VERTEX_BUFFER_VIEW VertexBufferView() const {
        D3D12_VERTEX_BUFFER_VIEW vbv;
        vbv.BufferLocation = VertexBufferGPU->GetGPUVirtualAddress();
        vbv.StrideInBytes = VertexByteStride;
        vbv.SizeInBytes = VertexBufferByteSize;
        return vbv;
    }

    /*
     * @brief Get the index buffer view.
     */
    D3D12_INDEX_BUFFER_VIEW IndexBufferView() const {
        D3D12_INDEX_BUFFER_VIEW ibv;
        ibv.BufferLocation = IndexBufferGPU->GetGPUVirtualAddress();
        ibv.Format = IndexFormat;
        ibv.SizeInBytes = IndexBufferByteSize;
        return ibv;
    }

    /*
     * @brief Upload geometry data and set necessary information about the geometry.
     */
    void Create(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList, std::unique_ptr<Geometry> &geoData) {
        if (device == nullptr) {
            throw std::runtime_error("device is nullptr");
        }
        if (cmdList == nullptr) {
            throw std::runtime_error("cmdList is nullptr");
        }
        if (geoData == nullptr) {
            throw std::runtime_error("geoData is nullptr");
        }
        auto geometry = geoData.get();
        ThrowIfFailed(D3DCreateBlob(geometry->vertexByteSize, &this->VertexBufferCPU));
        CopyMemory(this->VertexBufferCPU->GetBufferPointer(), geometry->vertexData.get(), geometry->vertexByteSize);

        ThrowIfFailed(D3DCreateBlob(geometry->indexByteSize, &this->IndexBufferCPU));
        CopyMemory(this->IndexBufferCPU->GetBufferPointer(), geometry->indexData.data(), geometry->indexByteSize);

        this->VertexBufferGPU = CreateDefaultBuffer(device, cmdList, geometry->vertexData.get(),
                                                    geometry->vertexByteSize, this->VertexBufferUploader);

        this->IndexBufferGPU = CreateDefaultBuffer(device, cmdList, geometry->indexData.data(), geometry->indexByteSize,
                                                   this->IndexBufferUploader);

        this->VertexByteStride = geometry->vertexByteStride;
        this->VertexBufferByteSize = geometry->vertexByteSize;
        this->IndexBufferByteSize = geometry->indexByteSize;
        this->IndexCount = geometry->indexNum;
    }
};
