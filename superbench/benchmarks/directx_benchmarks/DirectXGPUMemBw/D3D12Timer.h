// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <d3d12.h>

namespace D3D12 {
struct GPUTimestampPair {
    UINT64 Start;
    UINT64 Stop;
};

// D3D12 timer.
class D3D12Timer {
  public:
    // Constructor.
    D3D12Timer();

    // Destructor.
    ~D3D12Timer();

    HRESULT init(ID3D12Device *pDevice, UINT numTimers);

    // Start timestamp.
    void start(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex);

    // Stop timestamp.
    void stop(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex);

    // Resolve query data. Write query to device memory. Make sure to wait for query to finsih before resolving data.
    void resolveQueryToCPU(ID3D12GraphicsCommandList *pCommandList, UINT timestampPairIndex);

    // Get start and end timestamp pair.
    GPUTimestampPair getTimestampPair(UINT timestampPairIndex);

    // Whether timer is active.
    bool isActive();

  private:
    ID3D12Device *device_ = nullptr;
    ID3D12QueryHeap *queryHeap_ = nullptr;
    ID3D12Resource *queryResourceCPU_ = nullptr;
    bool active_ = false;
    UINT timerCount_ = 0;
};
} // namespace D3D12
