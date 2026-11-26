// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "common.h"
#include "inline_common.h"
#include "memcpy.h"
#include "output.h"
#include "testcase.h"

// Inherit the native all-host filters, including the bounce-buffer CC waiver.
class AllToHostBatchCE : public AllToHostTransfer {
  public:
    AllToHostBatchCE() : AllToHostTransfer(CopyInitiator::CE) {
        key = "all_to_host_batch_memcpy_ce";
        desc = "\tMeasures bandwidth of cuMemcpyAsync from all devices to host simultaneously.\n"
               "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
               "\tIndividual device bandwidths are measured and reported separately.";
    }
    void run(unsigned long long size, unsigned long long loopCount) override;
};

class HostToAllBatchCE : public HostToAllTransfer {
  public:
    HostToAllBatchCE() : HostToAllTransfer(CopyInitiator::CE) {
        key = "host_to_all_batch_memcpy_ce";
        desc = "\tMeasures bandwidth of cuMemcpyAsync from host to all devices simultaneously.\n"
               "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
               "\tIndividual device bandwidths are measured and reported separately.";
    }
    void run(unsigned long long size, unsigned long long loopCount) override;
};

class AllToHostBatchSM : public AllToHostTransfer {
  public:
    AllToHostBatchSM() : AllToHostTransfer(CopyInitiator::SM) {
        key = "all_to_host_batch_memcpy_sm";
        desc = "\tMeasures bandwidth of copy kernels from all devices to host simultaneously.\n"
               "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
               "\tIndividual device bandwidths are measured and reported separately.";
    }
    void run(unsigned long long size, unsigned long long loopCount) override;
};

class HostToAllBatchSM : public HostToAllTransfer {
  public:
    HostToAllBatchSM() : HostToAllTransfer(CopyInitiator::SM) {
        key = "host_to_all_batch_memcpy_sm";
        desc = "\tMeasures bandwidth of copy kernels from host to all devices simultaneously.\n"
               "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
               "\tIndividual device bandwidths are measured and reported separately.";
    }
    void run(unsigned long long size, unsigned long long loopCount) override;
};

void Testcase::allHostHelperBatch(unsigned long long size, MemcpyOperation &memcpyInstance,
                                  PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost) {
    std::vector<std::unique_ptr<MemcpyBuffer>> hostBuffers;
    std::vector<std::unique_ptr<MemcpyBuffer>> deviceBuffers;
    std::vector<const MemcpyBuffer *> allSrcBuffers;
    std::vector<const MemcpyBuffer *> allDstBuffers;

    // Keep an equal-sized buffer pair per GPU alive for one native concurrent operation.
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hostBuffers.push_back(std::make_unique<HostBuffer>(size, deviceId));
        deviceBuffers.push_back(std::make_unique<DeviceBuffer>(size, deviceId));
        allSrcBuffers.push_back(sourceIsHost ? hostBuffers.back().get() : deviceBuffers.back().get());
        allDstBuffers.push_back(sourceIsHost ? deviceBuffers.back().get() : hostBuffers.back().get());
    }

    // Native dispatch owns verification, timing, sampling and per-copy bandwidth calculation.
    std::vector<double> deviceBandwidths = memcpyInstance.doMemcpyVector(allSrcBuffers, allDstBuffers);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        bandwidthValues.value(0, deviceId) = deviceBandwidths[deviceId];
    }
    recordStability(bandwidthValues, memcpyInstance);
}

void AllToHostBatchCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    allHostHelperBatch(size, memcpyInstance, bandwidthValues, false);
    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) <- GPU(column) batch bandwidth (GB/s)");
}

void HostToAllBatchCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    allHostHelperBatch(size, memcpyInstance, bandwidthValues, true);
    output->addTestcaseResults(bandwidthValues, "memcpy CE CPU(row) -> GPU(column) batch bandwidth (GB/s)");
}

void AllToHostBatchSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    allHostHelperBatch(size, memcpyInstance, bandwidthValues, false);
    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <- GPU(column) batch bandwidth (GB/s)");
}

void HostToAllBatchSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    allHostHelperBatch(size, memcpyInstance, bandwidthValues, true);
    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) batch bandwidth (GB/s)");
}
