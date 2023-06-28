// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "D3D12Timer.h"
#include "../directx_third_party/DXSampleHelper.h"
#include "../directx_third_party/d3dx12.h"
#include <cassert>

namespace D3D12 {
D3D12Timer::D3D12Timer() {}

// Destructor.
D3D12Timer::~D3D12Timer() {
    if (m_queryHeap)
        m_queryHeap->Release();
    if (m_queryResourceCPU)
        m_queryResourceCPU->Release();
}

void D3D12Timer::init(ID3D12Device *pDevice, ID3D12CommandQueue *pCommandQueue, UINT numTimers, QueueType type) {
    assert(pDevice != nullptr);
    m_device = pDevice;
    m_timerCount = numTimers;

    UINT64 gpuFreq;
    ThrowIfFailed(pCommandQueue->GetTimestampFrequency(&gpuFreq));
    m_gpuFreqInv = 1000.0 / double(gpuFreq);

    D3D12_QUERY_HEAP_DESC queryHeapDesc;
    queryHeapDesc.Count = m_timerCount * 2;
    queryHeapDesc.NodeMask = 0;
    if (type == QueueType::compute) {
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    } else if (type == QueueType::copy) {
        queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP;
    }
    ThrowIfFailed(m_device->CreateQueryHeap(&queryHeapDesc, IID_PPV_ARGS(&m_queryHeap)));

    D3D12_HEAP_PROPERTIES heapProp = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    D3D12_RESOURCE_DESC resouceDesc = CD3DX12_RESOURCE_DESC::Buffer(m_timerCount * sizeof(GPUTimestampPair));
    ThrowIfFailed(m_device->CreateCommittedResource(&heapProp, D3D12_HEAP_FLAG_NONE, &resouceDesc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&m_queryResourceCPU)));
}

// Start timestamp.
bool D3D12Timer::start(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex) {
    if (timestampPairIndex >= m_timerCount)
        return false;
    pCommandList->EndQuery(m_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, getStartIndex(timestampPairIndex));
    return true;
}

// Stop timestamp.
bool D3D12Timer::stop(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex) {
    if (timestampPairIndex >= m_timerCount)
        return false;
    pCommandList->EndQuery(m_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, getEndIndex(timestampPairIndex));
    return true;
}

// Resolve query data. Write query to device memory. Make sure to wait for query to finish before resolving data.
void D3D12Timer::resolveQueryToCPU(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex) {
    pCommandList->ResolveQueryData(m_queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, getStartIndex(timestampPairIndex), 2,
                                   m_queryResourceCPU, sizeof(GPUTimestampPair) * timestampPairIndex);
}

// Get start and end timestamp pair.
double D3D12Timer::getElapsedMsByTimestampPair(UINT timestampPairIndex) {
    GPUTimestampPair *timingData = nullptr;
    D3D12_RANGE readRange{sizeof(GPUTimestampPair) * timestampPairIndex,
                          sizeof(GPUTimestampPair) * (timestampPairIndex + 1)};
    D3D12_RANGE writeRange{0, 0};
    if (SUCCEEDED(m_queryResourceCPU->Map(0, &readRange, (void **)&timingData))) {
        m_queryResourceCPU->Unmap(0, &writeRange);
        return (timingData->Stop - timingData->Start) * m_gpuFreqInv;
    }
    return -1;
}
} // namespace D3D12
