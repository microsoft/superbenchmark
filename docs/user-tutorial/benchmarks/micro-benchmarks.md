---
id: micro-benchmarks
---

# Micro Benchmarks

## Computation Benchmarks

### `kernel-launch`

#### Introduction

Measure GPU kernel launch latency,
which is defined as the time range from the beginning of the launch API call to the beginning of the kernel execution.

#### Metrics

| Name                         | Unit      | Description                          |
|------------------------------|-----------|--------------------------------------|
| kernel-launch/event_overhead | time (ms) | Launch latency measured in GPU time. |
| kernel-launch/wall_overhead  | time (ms) | Launch latency measured in CPU time. |

### `gemm-flops`

#### Introduction

Measure the GPU GEMM FLOPS for different float and int data types, with or without Tensor Core (XDLOPS),
performed by NVIDIA [cutlass](https://github.com/NVIDIA/cutlass/tree/ccb697bac77fcc898e9c897b2c90aa5b60ac72fb)
or AMD [rocblas-bench](https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/benchmarks).

#### Metrics

| Name                   | Unit           | Description                                             |
|------------------------|----------------|---------------------------------------------------------|
| gemm-flops/FP64        | FLOPS (GFLOPS) | GEMM float64 peak FLOPS.                                |
| gemm-flops/FP32        | FLOPS (GFLOPS) | GEMM float32 peak FLOPS.                                |
| gemm-flops/FP16        | FLOPS (GFLOPS) | GEMM float16 peak FLOPS.                                |
| gemm-flops/FP64_TC     | FLOPS (GFLOPS) | GEMM float64 peak FLOPS with NVIDIA Tensor Core.        |
| gemm-flops/TF32_TC     | FLOPS (GFLOPS) | GEMM tensor-float32 peak FLOPS with NVIDIA Tensor Core. |
| gemm-flops/FP16_TC     | FLOPS (GFLOPS) | GEMM float16 peak FLOPS with NVIDIA Tensor Core.        |
| gemm-flops/BF16_TC     | FLOPS (GFLOPS) | GEMM bfloat16 peak FLOPS with NVIDIA Tensor Core.       |
| gemm-flops/INT8_TC     | IOPS (GIOPS)   | GEMM int8 peak IOPS with NVIDIA Tensor Core.            |
| gemm-flops/INT4_TC     | IOPS (GIOPS)   | GEMM int4 peak IOPS with NVIDIA Tensor Core.            |
| gemm-flops/FP32_xDLOPS | FLOPS (GFLOPS) | GEMM tensor-float32 peak FLOPS with AMD XDLOPS.         |
| gemm-flops/FP16_xDLOPS | FLOPS (GFLOPS) | GEMM float16 peak FLOPS with AMD XDLOPS.                |
| gemm-flops/BF16_xDLOPS | FLOPS (GFLOPS) | GEMM bfloat16 peak FLOPS with AMD XDLOPS.               |
| gemm-flops/INT8_xDLOPS | IOPS (GIOPS)   | GEMM int8 peak IOPS with AMD XDLOPS.                    |

### `matmul`

#### Introduction

Large scale matmul operation using `torch.matmul` with one GPU.

#### Metrics

| Name                      | Unit      | Description                    |
|---------------------------|-----------|--------------------------------|
| pytorch-matmul/nosharding | time (ms) | Time of pure matmul operation. |

### `cublas-function`

TODO

### `cudnn-function`

TODO

### `tensorrt-inference`

#### Introduction

Inference PyTorch/ONNX models on NVIDIA GPUs with [TensorRT](https://developer.nvidia.com/tensorrt).

#### Metrics

| Name                                      | Unit      | Description                                                                                              |
|-------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------|
| tensorrt-inference/gpu_lat_ms_mean        | time (ms) | The mean GPU latency to execute the kernels for a query.                                                 |
| tensorrt-inference/gpu_lat_ms_99          | time (ms) | The 99th percentile GPU latency to execute the kernels for a query.                                      |
| tensorrt-inference/host_lat_ms_mean       | time (ms) | The mean H2D, GPU, and D2H latency to execute the kernels for a query.                                   |
| tensorrt-inference/host_lat_ms_99         | time (ms) | The 99th percentile H2D, GPU, and D2H latency to execute the kernels for a query.                        |
| tensorrt-inference/end_to_end_lat_ms_mean | time (ms) | The mean duration from when the H2D of a query is called to when the D2H of the same query is completed. |
| tensorrt-inference/end_to_end_lat_ms_99   | time (ms) | The P99 duration from when the H2D of a query is called to when the D2H of the same query is completed.  |

### `ort-inference`

#### Introduction

Inference performance of the torchvision models using ONNXRuntime.

#### Metrics

| Name                                          | Unit      | Description                                               |
|-----------------------------------------------|-----------|-----------------------------------------------------------|
| ort-inference/latency_{model}_{precision}     | time (ms) | The mean latency to execute one batch of inference.       |

## Communication Benchmarks

### `mem-bw`

#### Introduction

Measure the memory copy bandwidth across PCI-e and memory copy bandwidth between GPUs,
performed by [NVIDIA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/bandwidthTest)
or [AMD](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/1_Utils/hipBusBandwidth) bandwidth test tool.

#### Metrics

| Name              | Unit             | Description                      |
|-------------------|------------------|----------------------------------|
| mem-bw/H2D_Mem_BW | bandwidth (GB/s) | Host to device copy bandwidth.   |
| mem-bw/D2H_Mem_BW | bandwidth (GB/s) | Device to host copy bandwidth.   |
| mem-bw/D2D_Mem_BW | bandwidth (GB/s) | Device to device copy bandwidth. |

### `gpu-sm-copy-bw`

Measure the memory copy bandwidth across PCI-e and memory copy bandwidth between GPUs, initialized by GPU SM.

#### Metrics

| Name                | Unit             | Description                                          |
|---------------------|------------------|------------------------------------------------------|
| gpu-sm-copy-bw/htod | bandwidth (GB/s) | Host to device copy bandwidth initialized by GPU SM. |
| gpu-sm-copy-bw/dtoh | bandwidth (GB/s) | Device to host copy bandwidth initialized by GPU SM. |

### `ib-loopback`

#### Introduction

Measure the InfiniBand loopback verbs bandwidth, performed by
[OFED performance tests](https://github.com/linux-rdma/perftest/tree/7504ce48ac396a02f4d00de359257b2cb8458f06).

#### Metrics

| Name                                               | Unit             | Description                                                  |
|----------------------------------------------------|------------------|--------------------------------------------------------------|
| ib-loopback/IB\_write\_${msg\_size}\_Avg_${ib_dev} | bandwidth (MB/s) | InfiniBand loopback write bandwidth with given message size. |
| ib-loopback/IB\_read\_${msg\_size}\_Avg_${ib_dev}  | bandwidth (MB/s) | InfiniBand loopback read bandwidth with given message size.  |
| ib-loopback/IB\_send\_${msg\_size}\_Avg_${ib_dev}  | bandwidth (MB/s) | InfiniBand loopback send bandwidth with given message size.  |

### `nccl-bw` / `rccl-bw`

#### Introduction

Measure the performance of NCCL/RCCL operations,
performed by [nccl-tests](https://github.com/NVIDIA/nccl-tests/tree/44df0bf010dcc95e840ca0fb7466c67cff3f1f0f)
or [rccl-tests](https://github.com/ROCmSoftwarePlatform/rccl-tests/tree/dc1ad4853d7ec738387d42a75a58a98d7af00c7b).
Support the following operations currently: allreduce, allgather, broadcast, reduce, reducescatter, alltoall.

#### Metrics

| Name                                   | Unit             | Description                                                 |
|----------------------------------------|------------------|-------------------------------------------------------------|
| nccl-bw/${operation}_${msg_size}_time  | time (us)        | NCCL operation lantency with given message size.            |
| nccl-bw/${operation}_${msg_size}_algbw | bandwidth (GB/s) | NCCL operation algorithm bandwidth with given message size. |
| nccl-bw/${operation}_${msg_size}_busbw | bandwidth (GB/s) | NCCL operation bus bandwidth with given message size.       |
| rccl-bw/${operation}_${msg_size}_time  | time (us)        | RCCL operation lantency with given message size.            |
| rccl-bw/${operation}_${msg_size}_algbw | bandwidth (GB/s) | RCCL operation algorithm bandwidth with given message size. |
| rccl-bw/${operation}_${msg_size}_busbw | bandwidth (GB/s) | RCCL operation bus bandwidth with given message size.       |

### `tcp-connectivity`

#### Introduction

Test the TCP connectivity between current node and nodes in the hostfile,
performed by [tcping](https://github.com/zhengxiaowai/tcping)

#### Metrics

| Metrics                                      | Unit     | Description                                                                          |
| -------------------------------------------- | -------- | ------------------------------------------------------------------------------------ |
| tcp-connectivity/Successed_${hostname/ip}    | count    | successed times of tcp connections between current node and other nodes              |
| tcp-connectivity/Failed_${hostname/ip}       | count    | failed times of tcp connections between current node and other nodes                 |
| tcp-connectivity/Success_Rate_${hostname/ip} | count    | success rate(successed/total) of tcp connection between current node and other nodes |
| tcp-connectivity/Minimum_${hostname/ip}      | time(ms) | mininum latency of tcp connections between current node and other nodes              |
| tcp-connectivity/Maximum_${hostname/ip}      | time(ms) | maximum latency of tcp connections between current node and other nodes              |
| tcp-connectivity/Average_${hostname/ip}      | time(ms) | average latency of tcp connections between current node and other nodes              |

### `gpcnet-network-test` / `gpcnet-network-load-test`

#### Introduction

Distributed test, test the global network performance and congestion,
performed by [GPCNET](https://github.com/netbench/GPCNET)

gpcnet-network-test: Full system network tests in random and natural ring, alltoall and allreduce, at least 2 nodes

gpcnet-network-load-test: Select full system network tests run with four congestors to measure network congestion or contention, at least 10 nodes

 - test title: Isolated Network Tests, Isolated Congestion Tests, Network Tests running with Congestion Tests ( RR Two-sided Lat Network Test), Network Tests running with Congestion Tests (RR Two-sided BW+Sync Network Test), Network Tests running with Congestion Tests ( Multiple Allreduce Network Test), Network Tests running with Congestion Tests - Key Results
 - supporting network tests: RR Two-sided Lat (8 B), RR Two-sided BW+Sync (131072 B), Multiple Allreduce (8 B)
 - supporting congetors: Alltoall (4096 B), Two-sided Incast (4096 B), Put Incast (4096 B), Get Bcast (4096 B)

#### Metrics

| Metrics                                                             | Unit                  | Description                                                                                                                                                                |
| ------------------------------------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {benchmark_name}/${test_title}_RRTwo-sidedLat(8B)_${stat}           | time(usec)            | statistical values(Min, Max, Avg, 99%, 99.9%) obtained by all nodes use algorithm 'random ring communication pattern two-side latency' for network testing                 |
| {benchmark_name}/${test_title}_RRTwo-sidedBW+Sync(131072B)_${stat}  | MiB/s/rank            | fstatistical values(Min, Max, Avg, 99%, 99.9%) obtained by all nodes use algorithm 'random ring communication pattern two-side bandwidth with barrier' for network testing |
| {benchmark_name}/${test_title}_MultipleAllreduce(8B)_${stat}        | time(usec)            | statistical values(Min, Max, Avg, 99%, 99.9%) obtained by all nodes use algorithm 'multiple allreduce bandwidth' for network testing                                       |
| {benchmark_name}/${test_title}_GetBcast(4096B)_${stat}              | bandwidth (MB/s/rank) | statistical values(Min, Max, Avg, 99%, 99.9%) obtained by all nodes use congestion 'Get Bcast(4096B)' for congestion testing                                               |
| {benchmark_name}/${test_title}_PutIncast(4096B)_${stat}             | bandwidth (MB/s/rank) | statistical values(Min, Max, Avg, 99%, 99.9%) obtained by all nodes use congestion 'Put Incast (4096 B)' for congestion testing                                            |
| {benchmark_name}/${test_title}_Two-sidedIncast(4096B)_${stat}       | bandwidth (MB/s/rank) | statistical values(Min, Max, Avg, 99%, 99.9%) obtained by all nodes use congestion 'Two-sided Incast (4096 B)' for congestion testing                                      |
| {benchmark_name}/${test_title}_Alltoall(4096B)_${stat}              | bandwidth (MB/s/rank) | statistical values(Min, Max, Avg, 99%, 99.9%) obtained by all nodes use congestion 'Alltoall (4096 B)' for congestion testing                                              |
| gpcnet-network-load-test/${test_title}_${network_test_algo}_${stat} | times(x)              | summary about congestion impact factor of every network test algorithm                                                                                                     |

## Computation-communication Benchmarks

### `computation-communication-overlap`

#### Introduction

Test the performance of single node when communication and computation overlap.

#### Metrics

| Name                                                  | Unit      | Description                                                  |
|-------------------------------------------------------|-----------|--------------------------------------------------------------|
| pytorch-computation-communication-overlap/mul_cost    | time (ms) | Time of communication and mul kernel computation overlap.    |
| pytorch-computation-communication-overlap/matmul_cost | time (ms) | Time of communication and matmul kernel computation overlap. |

####

### `sharding-matmul`

#### Introduction

Test the performance of large scale matmul operation with multiple GPUs:
* allreduce: Each GPU will calculate part of the MM calculation, and use AllReduce to merge all data into one tensor.
* allgather: Each GPU will calculate part of the MM calculation, and use AllGather + Concat to merge all data into one tensor.

#### Metrics

| Name                              | Unit      | Description                              |
|-----------------------------------|-----------|------------------------------------------|
| pytorch-sharding-matmul/allreduce | time (ms) | Time of sharding matmul using allreduce. |
| pytorch-sharding-matmul/allgather | time (ms) | Time of sharding matmul using allgather. |

## Storage Benchmarks

### `disk-benchmark`

#### Introduction

Measure the disk performance through [FIO](https://github.com/axboe/fio/tree/0313e938c9c8bb37d71dade239f1f5326677b079).

#### Metrics

| Name                                                               | Unit         | Description                                              |
|--------------------------------------------------------------------|--------------|----------------------------------------------------------|
| disk-benchmark/${disk_name}_rand_read_write_bs                     | size (bytes) | Disk random read write block size.                       |
| disk-benchmark/${disk_name}_rand_read_write_read_iops              | IOPS         | Disk random read write read IOPS.                        |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_95.000000  | time (ns)    | Disk random read write read latency in 95.0 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_99.000000  | time (ns)    | Disk random read write read latency in 99.0 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_99.900000  | time (ns)    | Disk random read write read latency in 99.9 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_write_iops             | IOPS         | Disk random read write write IOPS.                       |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_95.000000 | time (ns)    | Disk random read write write latency in 95.0 percentile. |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_99.000000 | time (ns)    | Disk random read write write latency in 99.0 percentile. |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_99.900000 | time (ns)    | Disk random read write write latency in 99.9 percentile. |
