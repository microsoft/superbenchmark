// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <random>

#include "../directx_third_party/DXSampleHelper.h"
#include "../directx_third_party/d3dx12.h"

// Helper class for creating and uploading resources to the GPU.
template <typename T> class UploadBuffer {
  public:
    UploadBuffer(ID3D12Device *device, UINT elementCount, bool isConstantBuffer)
        : m_isConstantBuffer(isConstantBuffer) {
        m_elementByteSize = sizeof(T);

        if (isConstantBuffer)
            m_elementByteSize = CalcConstantBufferByteSize(sizeof(T));

        ThrowIfFailed(
            device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                                            &CD3DX12_RESOURCE_DESC::Buffer(m_elementByteSize * elementCount),
                                            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_uploadBuffer)));
    }

    UploadBuffer(const UploadBuffer &rhs) = delete;
    UploadBuffer &operator=(const UploadBuffer &rhs) = delete;
    ~UploadBuffer() {
        if (m_uploadBuffer != nullptr)
            m_uploadBuffer->Unmap(0, nullptr);

        m_mappedData = nullptr;
    }

    ID3D12Resource *Resource() const { return m_uploadBuffer.Get(); }

    void CopyData(int elementIndex, const T &data) {
        ThrowIfFailed(m_uploadBuffer->Map(0, nullptr, reinterpret_cast<void **>(&m_mappedData)));
        memcpy(&m_mappedData[elementIndex * m_elementByteSize], &data, sizeof(T));
        m_uploadBuffer->Unmap(0, nullptr);
    }

  private:
    Microsoft::WRL::ComPtr<ID3D12Resource> m_uploadBuffer;
    BYTE *m_mappedData = nullptr;

    UINT m_elementByteSize = 0;
    bool m_isConstantBuffer = false;
};

/*
 * @brief: Create a default buffer.
 * @param: device the device of GPU object.
 * @param: cmdList the command list of GPU object.
 * @param: initData the data to be copied to the default buffer.
 * @param: byteSize the size of data.
 * @return: the default buffer.
 */
Microsoft::WRL::ComPtr<ID3D12Resource> CreateDefaultBuffer(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList,
                                                           const void *initData, UINT64 byteSize,
                                                           Microsoft::WRL::ComPtr<ID3D12Resource> &uploadBuffer);

/*
 * @brief: Calculate the size of constant buffer.
 */
UINT CalcConstantBufferByteSize(UINT byteSize);

/*
 * @brief: Create a random texture.
 * @param: width the width of texture.
 * @param: height the height of texture.
 * @param: texturePixelSize the size of texture pixel.
 * @return: the random texture data.
 */
std::vector<UINT8> CreateRandomTexture(const UINT width, const UINT height, const UINT texturePixelSize = 4);

/*
 * @brief: Upload the texture to GPU.
 * @param: device the device of GPU object.
 * @param: pCmdList the command list of GPU object.
 * @param: textureData the texture data to be uploaded.
 * @param: texture the texture resource.
 * @param: width the width of texture.
 * @param: height the height of texture.
 * @param: texturePixelSize the size of texture pixel.
 */
void UploadTexture(ID3D12Device *device, ID3D12GraphicsCommandList *pCmdList, const std::vector<UINT8> &textureData,
                   Microsoft::WRL::ComPtr<ID3D12Resource> &texture, const UINT width, const UINT height,
                   const UINT texturePixelSize = 4);

/*
 * @brief: Create a texture resource.
 * @param: device the device of GPU object.
 * @param: width the width of texture.
 * @param: height the height of texture.
 * @param: format the format of texture.
 * @param: textureResource the texture resource.
 * @param: arraySize the size of texture array.
 */
void CreateTextureResource(ID3D12Device *device, UINT width, UINT height, DXGI_FORMAT format,
                           Microsoft::WRL::ComPtr<ID3D12Resource> &textureResource, UINT16 arraySize);

/*
 * @brief: Create a random texture resource and upload it to GPU.
 * @param: device the device of GPU object.
 * @param: cmdList the command list of GPU object.
 * @param: textureResource the texture resource.
 * @param: width the width of texture.
 * @param: height the height of texture.
 * @param: format the format of texture.
 */
void Texture2D(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList,
               Microsoft::WRL::ComPtr<ID3D12Resource> &textureResource, int width, int height, DXGI_FORMAT format);

/*
 * @brief: Create a random texture cube resource and upload it to GPU.
 * @param: device the device of GPU object.
 * @param: cmdList the command list of GPU object.
 * @param: textureResource the texture resource.
 * @param: width the width of texture.
 * @param: height the height of texture.
 * @param: format the format of texture.
 */
void TextureCube(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList,
                 Microsoft::WRL::ComPtr<ID3D12Resource> &textureResource, int width, int height, DXGI_FORMAT format);
