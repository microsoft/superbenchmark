// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.h"
#include "inline_common.h"
#include "output.h"
#include "testcase.h"

// All to Host Batch CE memcpy using cuMemcpyAsync
class AllToHostBatchCE : public Testcase {
  public:
    AllToHostBatchCE()
        : Testcase("all_to_host_batch_memcpy_ce",
                   "\tMeasures bandwidth of cuMemcpyAsync from all devices to host simultaneously.\n"
                   "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
                   "\tIndividual device bandwidths are measured and reported separately.") {}
    virtual ~AllToHostBatchCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to All Batch CE memcpy using cuMemcpyAsync
class HostToAllBatchCE : public Testcase {
  public:
    HostToAllBatchCE()
        : Testcase("host_to_all_batch_memcpy_ce",
                   "\tMeasures bandwidth of cuMemcpyAsync from host to all devices simultaneously.\n"
                   "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
                   "\tIndividual device bandwidths are measured and reported separately.") {}
    virtual ~HostToAllBatchCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// All to Host Batch SM memcpy using a copy kernel
class AllToHostBatchSM : public Testcase {
  public:
    AllToHostBatchSM()
        : Testcase("all_to_host_batch_memcpy_sm",
                   "\tMeasures bandwidth of copy kernels from all devices to host simultaneously.\n"
                   "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
                   "\tIndividual device bandwidths are measured and reported separately.") {}
    virtual ~AllToHostBatchSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to All Batch SM memcpy using a copy kernel
class HostToAllBatchSM : public Testcase {
  public:
    HostToAllBatchSM()
        : Testcase("host_to_all_batch_memcpy_sm",
                   "\tMeasures bandwidth of copy kernels from host to all devices simultaneously.\n"
                   "\tAll devices perform memcpy operations concurrently with the same buffer size.\n"
                   "\tIndividual device bandwidths are measured and reported separately.") {}
    virtual ~HostToAllBatchSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

void Testcase::allHostHelperBatch(unsigned long long size, MemcpyOperation &memcpyInstance,
                                  PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost) {
    std::vector<const MemcpyBuffer *> allSrcBuffers;
    std::vector<const MemcpyBuffer *> allDstBuffers;

    // Create buffers for all devices with the same size
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        if (sourceIsHost) {
            allSrcBuffers.push_back(new HostBuffer(size, deviceId));
            allDstBuffers.push_back(new DeviceBuffer(size, deviceId));
        } else {
            allSrcBuffers.push_back(new DeviceBuffer(size, deviceId));
            allDstBuffers.push_back(new HostBuffer(size, deviceId));
        }
    }

    // Perform memcpy for all devices in a single run and get individual bandwidths
    std::vector<double> deviceBandwidths = memcpyInstance.doMemcpyVector(allSrcBuffers, allDstBuffers);

    // Store individual bandwidth for each device
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        bandwidthValues.value(0, deviceId) = deviceBandwidths[deviceId];
    }

    // Clean up all buffers
    for (auto node : allSrcBuffers) {
        delete node;
    }

    for (auto node : allDstBuffers) {
        delete node;
    }
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
