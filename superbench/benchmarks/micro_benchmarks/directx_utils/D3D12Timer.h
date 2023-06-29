// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <d3d12.h>

namespace D3D12 {
struct GPUTimestampPair {
    UINT64 Start;
    UINT64 Stop;
};

enum QueueType { compute = 0, copy = 1 };

// D3D12 timer.
class D3D12Timer {
  public:
    // Constructor.
    D3D12Timer();

    // Destructor.
    ~D3D12Timer();

    void init(ID3D12Device *pDevice, ID3D12CommandQueue *pCommandQueue, UINT numTimers, QueueType type);

    // Start timestamp.
    bool start(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex);

    // Stop timestamp.
    bool stop(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex);

    // Resolve query data. Write query to device memory. Make sure to wait for query to finsih before resolving data.
    void resolveQueryToCPU(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex);

    // Get start and end timestamp pair.
    double getElapsedMsByTimestampPair(UINT timestampPairIndex);

    // Get the GPU frequency.
    double getGPUFrequecy() { return m_gpuFreqInv; }

    // Get start index of the selected timestamp pair
    UINT getStartIndex(UINT timestampPairIndex) { return timestampPairIndex * 2; }

    // Get end index of the selected timestamp pair
    UINT getEndIndex(UINT timestampPairIndex) { return timestampPairIndex * 2 + 1; }

  private:
    ID3D12Device *m_device = nullptr;
    ID3D12QueryHeap *m_queryHeap = nullptr;
    ID3D12Resource *m_queryResourceCPU = nullptr;
    UINT m_timerCount = 0;
    double m_gpuFreqInv;
};
} // namespace D3D12
