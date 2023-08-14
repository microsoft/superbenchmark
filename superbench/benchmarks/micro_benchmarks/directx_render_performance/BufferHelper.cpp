// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "BufferHelper.h"

// Function to calculate the byte size of the constant buffer,
// which must be a multiple of 256 bytes.
UINT CalcConstantBufferByteSize(UINT byteSize) {
    // Calculate the aligned size.
    return (byteSize + 255) & ~255;
}

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
                                                           Microsoft::WRL::ComPtr<ID3D12Resource> &uploadBuffer) {
    ComPtr<ID3D12Resource> defaultBuffer;

    // Create the actual default buffer resource.
    ThrowIfFailed(device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                                                  D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(byteSize),
                                                  D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  IID_PPV_ARGS(defaultBuffer.GetAddressOf())));

    // In order to copy CPU memory data into our default buffer, we need to create
    // an intermediate upload heap.
    ThrowIfFailed(device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                                                  D3D12_HEAP_FLAG_NONE, &CD3DX12_RESOURCE_DESC::Buffer(byteSize),
                                                  D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                  IID_PPV_ARGS(uploadBuffer.GetAddressOf())));

    // Describe the data we want to copy into the default buffer.
    D3D12_SUBRESOURCE_DATA subResourceData = {};
    subResourceData.pData = initData;
    subResourceData.RowPitch = byteSize;
    subResourceData.SlicePitch = subResourceData.RowPitch;

    // Schedule to copy the data to the default buffer resource.  At a high level, the helper function
    // UpdateSubresources will copy the CPU memory into the intermediate upload heap.  Then, using
    // ID3D12CommandList::CopySubresourceRegion, the intermediate upload heap data will be copied to mBuffer.
    cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(), D3D12_RESOURCE_STATE_COMMON,
                                                                      D3D12_RESOURCE_STATE_COPY_DEST));
    UpdateSubresources<1>(cmdList, defaultBuffer.Get(), uploadBuffer.Get(), 0, 0, 1, &subResourceData);
    cmdList->ResourceBarrier(1,
                             &CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                                   D3D12_RESOURCE_STATE_GENERIC_READ));

    // Note: uploadBuffer has to be kept alive after the above function calls because
    // the command list has not been executed yet that performs the actual copy.
    // The caller can Release the uploadBuffer after it knows the copy has been executed.

    return defaultBuffer;
}

std::vector<UINT8> CreateRandomTexture(const UINT width, const UINT height, const UINT texturePixelSize) {
    // Create a buffer to store the texture data
    std::vector<unsigned char> textureData(width * height * texturePixelSize);

    // Initialize the random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, 255);

    // Generate random data for the texture
    for (UINT i = 0; i < width * height * texturePixelSize; ++i) {
        textureData[i] = static_cast<unsigned char>(distribution(generator));
    }
    return textureData;
}

void UploadTexture(ID3D12Device *device, ID3D12GraphicsCommandList *pCmdList, const std::vector<UINT8> &textureData,
                   Microsoft::WRL::ComPtr<ID3D12Resource> &texture, const UINT width, const UINT height,
                   const UINT texturePixelSize) {
    // Create the GPU upload buffer.
    const UINT64 uploadBufferSize = GetRequiredIntermediateSize(texture.Get(), 0, 1);

    ID3D12Resource *textureUploadHeap;
    ThrowIfFailed(
        device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
                                        &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&textureUploadHeap)));

    // Copy data to the intermediate upload heap and then schedule a copy
    // from the upload heap to the Texture2D.
    D3D12_SUBRESOURCE_DATA textureDataDesc = {};
    textureDataDesc.pData = textureData.data();
    textureDataDesc.RowPitch = width * texturePixelSize;
    textureDataDesc.SlicePitch = textureDataDesc.RowPitch * height;

    UpdateSubresources(pCmdList, texture.Get(), textureUploadHeap, 0, 0, 1, &textureDataDesc);
    pCmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
}

void CreateTextureResource(ID3D12Device *device, UINT width, UINT height, DXGI_FORMAT format,
                           Microsoft::WRL::ComPtr<ID3D12Resource> &textureResource, UINT16 arraySize) {
    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.DepthOrArraySize = arraySize;
    textureDesc.MipLevels = 1;
    textureDesc.Format = format;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ThrowIfFailed(device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                                                  D3D12_HEAP_FLAG_NONE, &textureDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                  nullptr, IID_PPV_ARGS(&textureResource)));
}

void Texture2D(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList,
               Microsoft::WRL::ComPtr<ID3D12Resource> &textureResource, int width, int height, DXGI_FORMAT format) {
    CreateTextureResource(device, width, height, format, textureResource, 1);
    auto textureData = CreateRandomTexture(width, height);
    UploadTexture(device, cmdList, textureData, textureResource, width, height);
}

void TextureCube(ID3D12Device *device, ID3D12GraphicsCommandList *cmdList,
                 Microsoft::WRL::ComPtr<ID3D12Resource> &textureResource, int width, int height, DXGI_FORMAT format) {
    CreateTextureResource(device, width, height, format, textureResource, 6);
    std::vector<UINT8> textureCubeData;
    for (int i = 0; i < 6; ++i) {
        auto textureData = CreateRandomTexture(width, height);
        textureCubeData.insert(textureCubeData.end(), textureData.begin(), textureData.end());
    }
    UploadTexture(device, cmdList, textureCubeData, textureResource, width, height);
}
