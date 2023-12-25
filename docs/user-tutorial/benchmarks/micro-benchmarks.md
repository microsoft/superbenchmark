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

| Name                     | Unit      | Description                          |
|--------------------------|-----------|--------------------------------------|
| kernel-launch/event_time | time (ms) | Launch latency measured in GPU time. |
| kernel-launch/wall_time  | time (ms) | Launch latency measured in CPU time. |

### `gemm-flops`

#### Introduction

Measure the GPU GEMM FLOPS for different float and int data types, with or without Tensor Core (XDLOPS),
performed by NVIDIA [cutlass](https://github.com/NVIDIA/cutlass/tree/ccb697bac77fcc898e9c897b2c90aa5b60ac72fb)
or AMD [rocblas-bench](https://github.com/ROCmSoftwarePlatform/rocBLAS/tree/develop/clients/benchmarks).

#### Metrics

| Name                         | Unit           | Description                                             |
|------------------------------|----------------|---------------------------------------------------------|
| gemm-flops/fp64_flops        | FLOPS (GFLOPS) | GEMM float64 peak FLOPS.                                |
| gemm-flops/fp32_flops        | FLOPS (GFLOPS) | GEMM float32 peak FLOPS.                                |
| gemm-flops/fp16_flops        | FLOPS (GFLOPS) | GEMM float16 peak FLOPS.                                |
| gemm-flops/fp64_tc_flops     | FLOPS (GFLOPS) | GEMM float64 peak FLOPS with NVIDIA Tensor Core.        |
| gemm-flops/tf32_tc_flops     | FLOPS (GFLOPS) | GEMM tensor-float32 peak FLOPS with NVIDIA Tensor Core. |
| gemm-flops/fp16_tc_flops     | FLOPS (GFLOPS) | GEMM float16 peak FLOPS with NVIDIA Tensor Core.        |
| gemm-flops/bf16_tc_flops     | FLOPS (GFLOPS) | GEMM bfloat16 peak FLOPS with NVIDIA Tensor Core.       |
| gemm-flops/int8_tc_iops      | IOPS (GIOPS)   | GEMM int8 peak IOPS with NVIDIA Tensor Core.            |
| gemm-flops/int4_tc_iops      | IOPS (GIOPS)   | GEMM int4 peak IOPS with NVIDIA Tensor Core.            |
| gemm-flops/fp32_xdlops_flops | FLOPS (GFLOPS) | GEMM tensor-float32 peak FLOPS with AMD XDLOPS.         |
| gemm-flops/fp16_xdlops_flops | FLOPS (GFLOPS) | GEMM float16 peak FLOPS with AMD XDLOPS.                |
| gemm-flops/bf16_xdlops_flops | FLOPS (GFLOPS) | GEMM bfloat16 peak FLOPS with AMD XDLOPS.               |
| gemm-flops/int8_xdlops_iops  | IOPS (GIOPS)   | GEMM int8 peak IOPS with AMD XDLOPS.                    |

### `matmul`

#### Introduction

Large scale matmul operation using `torch.matmul` with one GPU.

#### Metrics

| Name                           | Unit      | Description                    |
|--------------------------------|-----------|--------------------------------|
| pytorch-matmul/nosharding_time | time (ms) | Time of pure matmul operation. |

### `cublaslt-gemm` / `hipblaslt-gemm`

#### Introduction

Measure the GEMM performance of [`cublasLtMatmul`](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul) or [`hipblasLt-bench`](https://github.com/ROCm/hipBLASLt/blob/develop/clients/benchmarks/README.md).

#### Metrics

| Name                                                      | Unit           | Description                     |
|-----------------------------------------------------------|----------------|---------------------------------|
| cublaslt-gemm/${dtype}\_${batch}\_${m}\_${n}\_${k}_flops  | FLOPS (TFLOPS) | TFLOPS of measured GEMM kernel. |
| hipblaslt-gemm/${dtype}\_${batch}\_${m}\_${n}\_${k}_flops | FLOPS (TFLOPS) | TFLOPS of measured GEMM kernel. |

### `cublas-function`

#### Introduction

Measure the performance of most common Nvidia cuBLAS functions with parameters in models training including ResNet, VGG, DenseNet, LSTM, BERT, and GPT-2.

The supported functions for cuBLAS are as follows:
 - cublasSgemm
 - cublasSgemmStridedBatched
 - cublasGemmStridedBatchedEx
 - cublasGemmEx
 - cublasCgemm3mStridedBatched
 - cublasCgemm

#### Metrics

| Name                                                              | Unit      | Description                                                                                                                                  |
|-------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| cublas-function/name\_${function_name}\_${parameters}_time        | time (us) | The mean time to execute the cublas function with the parameters.                                                                            |
| cublas-function/name\_${function_name}\_${parameters}_correctness |           | Whether the calculation results of executing the cublas function with the parameters pass the correctness check if enable correctness check. |
| cublas-function/name\_${function_name}\_${parameters}_error       |           | The error ratio of the calculation results of executing the cublas function with the parameters if enable correctness check.                 |

### `cudnn-function`

#### Introduction

Measure the performance of most common Nvidia cuDNN functions with parameters in models training including ResNet, VGG, DenseNet, LSTM, BERT, and GPT-2.

The supported functions for cuDNN are as follows:
 - cudnnConvolutionBackwardFilter
 - cudnnConvolutionBackwardData
 - cudnnConvolutionForward

#### Metrics

| Name                                                      | Unit      | Description                                                      |
|-----------------------------------------------------------|-----------|------------------------------------------------------------------|
| cudnn-function/name\_${function_name}\_${parameters}_time | time (us) | The mean time to execute the cudnn function with the parameters. |

### `tensorrt-inference`

#### Introduction

Inference PyTorch/ONNX models on NVIDIA GPUs with [TensorRT](https://developer.nvidia.com/tensorrt).

Currently the following models are supported:

> alexnet, densenet121, densenet169, densenet201, densenet161, googlenet, inception_v3, mnasnet0_5,
> mnasnet1_0, mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d,
> resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, shufflenet_v2_x0_5, shufflenet_v2_x1_0,
> squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19
> lstm, bert-base, bert-large, gpt2-small

> Do not support large models like `gpt2-large` currently because models larger than 2GB (maximum protobuf size) cannot be exported in one ONNX file.

#### Metrics

| Name                                             | Unit      | Description                                                                                              |
|--------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------|
| tensorrt-inference/${model}_gpu_time_mean        | time (ms) | The mean GPU latency to execute the kernels for a query.                                                 |
| tensorrt-inference/${model}_gpu_time_99          | time (ms) | The 99th percentile GPU latency to execute the kernels for a query.                                      |
| tensorrt-inference/${model}_host_time_mean       | time (ms) | The mean H2D, GPU, and D2H latency to execute the kernels for a query.                                   |
| tensorrt-inference/${model}_host_time_99         | time (ms) | The 99th percentile H2D, GPU, and D2H latency to execute the kernels for a query.                        |
| tensorrt-inference/${model}_end_to_end_time_mean | time (ms) | The mean duration from when the H2D of a query is called to when the D2H of the same query is completed. |
| tensorrt-inference/${model}_end_to_end_time_99   | time (ms) | The P99 duration from when the H2D of a query is called to when the D2H of the same query is completed.  |

### `ort-inference`

#### Introduction

Inference performance of the torchvision models using ONNXRuntime. Currently the following models are supported:

> alexnet, densenet121, densenet169, densenet201, densenet161, googlenet, inception_v3, mnasnet0_5,
> mnasnet1_0, mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d,
> resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, shufflenet_v2_x0_5, shufflenet_v2_x1_0,
> squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19

The supported percentiles are 50, 90, 95, 99, and 99.9.

#### Metrics

| Name                                                | Unit      | Description                                                              |
|-----------------------------------------------------|-----------|--------------------------------------------------------------------------|
| ort-inference/{precision}_{model}_time              | time (ms) | The mean latency to execute one batch of inference.                      |
| ort-inference/{precision}_{model}_time_{percentile} | time (ms) | The {percentile}th percentile latency to execute one batch of inference. |

### `gpu-burn`

#### Introduction

Multi-GPU CUDA stress test for GPU compute and memory utilization, performed by [gpu-burn](https://github.com/wilicc/gpu-burn).
Supports the use of double unit types and the use of tensor cores.

#### Metrics

| Name                    | Unit     | Description                                                                        |
|-------------------------|----------|------------------------------------------------------------------------------------|
| gpu-burn/time           | time (s) | The runtime for gpu-burn test.                                                     |
| gpu-burn/gpu_[0-9]_pass | yes/no   | The result of the gpu-burn test for each GPU (1: yes, 0: no).                      |
| gpu-burn/abort          | yes/no   | Whether or not GPU-burn test aborted before returning GPU results (1: yes, 0: no). |

### `cpu-hpl`

#### Introduction

HPL or High Performance Computing Linpack evaluates compute bandwidth by solving dense linear systems in double precision arethmetic.
Performed by [High-Performance Linpack Benchmark for Distributed-Memory Computers](https://netlib.org/benchmark/hpl/)

#### Metrics

| Name               | Unit               | Description                                                               |
|--------------------|--------------------|---------------------------------------------------------------------------|
| cpu-hpl/tests_pass |                    | HPL completed running and correctness test has passed (1: pass, 0: fail). |
| cpu-hpl/throughput | bandwidth (GFlops) | Compute bandwidth.                                                        |
| cpu-hpl/time       | time (s)           | Time elapsed during HPL run.                                              |

### `cpu-stream`

#### Introduction

Measure of memory bandwidth and computation rate for simple vector kernels.
performed by [University of Virginia STREAM benchmark](https://www.cs.virginia.edu/stream/ref.html).

#### Metrics

| Name                                                     | Unit             | Description                                                    |
|----------------------------------------------------------|------------------|----------------------------------------------------------------|
| cpu-stream/threads                                       |                  | Number of threads used for the test. Determined by core count. |
| cpu-stream/['copy', 'scale', 'add', 'triad']\_throughput | bandwidth (MB/s) | Memory throughput of designated kerel operation.               |
| cpu-stream/['copy', 'scale', 'add', 'triad']\_time_avg   | time (s)         | Average elapsed times over all iterations.                     |
| cpu-stream/['copy', 'scale', 'add', 'triad']\_time_min   | time (s)         | Minimum elapsed times over all iterations.                     |
| cpu-stream/['copy', 'scale', 'add', 'triad']\_time_max   | time (s)         | Maximum elapsed times over all iterations.                     |

## Communication Benchmarks

### `cpu-memory-bw-latency`

#### Introduction

Measure the memory copy bandwidth and latency across different CPU NUMA nodes.
performed by [Intel MLC Tool](https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html).

#### Metrics

| Name                                                                    | Unit             | Description                                                         |
|-------------------------------------------------------------------------|------------------|---------------------------------------------------------------------|
| cpu-memory-bw-latency/mem\_bandwidth\_matrix\_numa\_[0-9]+\_[0-9]+\_bw  | bandwidth (MB/s) | Former NUMA to latter NUMA memory bandwidth.                        |
| cpu-memory-bw-latency/mem\_bandwidth\_matrix\_numa\_[0-9]+\_[0-9]+\_lat | time (ns)        | Former NUMA to latter NUMA memory latency.                          |
| cpu-memory-bw-latency/mem\_max\_bandwidth\_all\_reads\_bw               | bandwidth (MB/s) | Whole-CPU maximum memory bandwidth, full read.                      |
| cpu-memory-bw-latency/mem\_max\_bandwidth\_3_1\_reads-writes\_bw        | bandwidth (MB/s) | Whole-CPU maximum memory bandwidth, read : write = 3 : 1.           |
| cpu-memory-bw-latency/mem\_max\_bandwidth\_2_1\_reads-writes\_bw        | bandwidth (MB/s) | Whole-CPU maximum memory bandwidth, read : write = 2 : 1.           |
| cpu-memory-bw-latency/mem\_max\_bandwidth\_1_1\_reads-writes\_bw        | bandwidth (MB/s) | Whole-CPU maximum memory bandwidth, read : write = 1 : 1.           |
| cpu-memory-bw-latency/mem\_max\_bandwidth\_stream-triad\_like\_bw       | bandwidth (MB/s) | Whole-CPU maximum memory bandwidth, with stream-triad like pattern. |

### `mem-bw`

#### Introduction

Measure the memory copy bandwidth across PCI-e and memory copy bandwidth between GPUs,
performed by [NVIDIA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/bandwidthTest)
or [AMD](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/1_Utils/hipBusBandwidth) bandwidth test tool.

#### Metrics

| Name          | Unit             | Description                      |
|---------------|------------------|----------------------------------|
| mem-bw/h2d_bw | bandwidth (GB/s) | Host to device copy bandwidth.   |
| mem-bw/d2h_bw | bandwidth (GB/s) | Device to host copy bandwidth.   |
| mem-bw/d2d_bw | bandwidth (GB/s) | Device to device copy bandwidth. |

### `gpu-copy-bw`

Measure the memory copy bandwidth performed by GPU SM/DMA engine, including device-to-host, host-to-device and device-to-device.
For measurements of peer-to-peer communication performance between AMD GPUs, GPU memory buffers are allocated in `hipDeviceMallocUncached` (previous `hipDeviceMallocFinegrained`) mode to maximize performance.

#### Metrics

| Name                                                        | Unit             | Description                                                                                                                              |
|-------------------------------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| cpu\_to\_gpu[0-9]+\_by\_(sm\|dma)\_under\_numa[0-9]+\_bw    | bandwidth (GB/s) | The unidirectional bandwidth of one GPU reading one NUMA node's host memory using DMA engine or GPU SM.                                  |
| gpu[0-9]+\_to\_cpu\_by\_(sm\|dma)\_under\_numa[0-9]+\_bw    | bandwidth (GB/s) | The unidirectional bandwidth of one GPU writing one NUMA node's host memory using DMA engine or GPU SM.                                  |
| gpu[0-9]+\_to\_gpu[0-9]+\_by\_(sm\|dma)\_bw                 | bandwidth (GB/s) | The unidirectional bandwidth of one GPU reading or writing self's memory using DMA engine or GPU SM.                                     |
| gpu[0-9]+\_to\_gpu[0-9]+\_(read\|write)\_by\_(sm\|dma)\_bw  | bandwidth (GB/s) | The unidirectional bandwidth of one GPU reading or writing peer GPU's memory using DMA engine or GPU SM with peer communication enabled. |
| cpu\_and\_gpu[0-9]+\_by\_(sm\|dma)\_under\_numa[0-9]+\_bw   | bandwidth (GB/s) | The bidirectional bandwidth of one GPU reading and writing one NUMA node's host memory using DMA engine or GPU SM.                       |
| gpu[0-9]+\_and\_cpu\_by\_(sm\|dma)\_under\_numa[0-9]+\_bw   | bandwidth (GB/s) | Same as above, but generated by --dtoh --bidirectional.                                                                                  |
| gpu[0-9]+\_and\_gpu[0-9]+\_by\_(sm\|dma)\_bw                | bandwidth (GB/s) | The bidirectional bandwidth of one GPU reading and writing self's memory using DMA engine or GPU SM.                                     |
| gpu[0-9]+\_and\_gpu[0-9]+\_(read\|write)\_by\_(sm\|dma)\_bw | bandwidth (GB/s) | The bidirectional bandwidth of one GPU reading and writing peer GPU's memory using DMA engine or GPU SM with peer communication enabled. |
| gpu[0-9]+\_to\_gpu\_all\_write\_by\_sm\_bw                  | bandwidth (GB/s) | The unidirectional bandwidth of one GPU writing all peer GPUs' memory using GPU SM with peer communication enabled.                      |
| gpu\_all\_to\_gpu[0-9]+\_write\_by\_sm\_bw                  | bandwidth (GB/s) | The unidirectional bandwidth of all peer GPUs writing one GPU's memory using GPU SM with peer communication enabled.                     |
| gpu\_all\_to\_gpu\_all\_write\_by\_sm\_bw                   | bandwidth (GB/s) | The unidirectional bandwidth of all peer GPUs writing all peer GPUs' memory using GPU SM with peer communication enabled.                |

### `ib-loopback`

#### Introduction

Measure the InfiniBand loopback verbs bandwidth, performed by
[OFED performance tests](https://github.com/linux-rdma/perftest/tree/7504ce48ac396a02f4d00de359257b2cb8458f06).

#### Metrics

| Name                                | Unit             | Description                                                  |
|-------------------------------------|------------------|--------------------------------------------------------------|
| ib-loopback/ib_write_bw_${msg_size} | bandwidth (GB/s) | InfiniBand loopback write bandwidth with given message size. |
| ib-loopback/ib_read_bw_${msg_size}  | bandwidth (GB/s) | InfiniBand loopback read bandwidth with given message size.  |
| ib-loopback/ib_send_bw_${msg_size}  | bandwidth (GB/s) | InfiniBand loopback send bandwidth with given message size.  |

### `nccl-bw` / `rccl-bw`

#### Introduction

Measure the performance of NCCL/RCCL operations under multi nodes' traffic pattern,
performed by [nccl-tests](https://github.com/NVIDIA/nccl-tests/tree/44df0bf010dcc95e840ca0fb7466c67cff3f1f0f)
or [rccl-tests](https://github.com/ROCmSoftwarePlatform/rccl-tests/tree/dc1ad4853d7ec738387d42a75a58a98d7af00c7b).
Support the following operations currently: allreduce, allgather, broadcast, reduce, reducescatter, alltoall.
Support both in-place and out-of-place measurements.

Support the following traffic patterns:
* `all-nodes`, validate the NCCL/RCCL performance across all VM nodes simultaneously.
* `pair-wise`, validate the NCCL/RCCL performance across VM pairs with all possible combinations in parallel.
* `k-batch`, validate the NCCL/RCCL performance across VM groups with a specified batch scale.
* `topo-aware`, validate the NCCL/RCCL performance across VM pairs with different distances/hops as a quick test.

#### Metrics

| Name                                   | Unit             | Description                                                 |
|----------------------------------------|------------------|-------------------------------------------------------------|
| nccl-bw/${operation}_${msg_size}_time  | time (us)        | NCCL operation lantency with given message size.            |
| nccl-bw/${operation}_${msg_size}_algbw | bandwidth (GB/s) | NCCL operation algorithm bandwidth with given message size. |
| nccl-bw/${operation}_${msg_size}_busbw | bandwidth (GB/s) | NCCL operation bus bandwidth with given message size.       |
| rccl-bw/${operation}_${msg_size}_time  | time (us)        | RCCL operation lantency with given message size.            |
| rccl-bw/${operation}_${msg_size}_algbw | bandwidth (GB/s) | RCCL operation algorithm bandwidth with given message size. |
| rccl-bw/${operation}_${msg_size}_busbw | bandwidth (GB/s) | RCCL operation bus bandwidth with given message size.       |

If mpi mode is enable and traffic pattern is specified, the metrics pattern will change to `nccl-bw/${operation}_${serial_index)_${parallel_index):${msg_size}_time`
- `serial_index` represents the serial index of the host group in serial.
- `parallel_index` represents the parallel index of the host list in parallel.

### `tcp-connectivity`

#### Introduction

Test the TCP connectivity between current node and nodes in the hostfile,
performed by [tcping](https://github.com/zhengxiaowai/tcping)

#### Metrics

| Metrics                                         | Unit      | Description                                                                           |
|-------------------------------------------------|-----------|---------------------------------------------------------------------------------------|
| tcp-connectivity/${hostname/ip}_successed_count | count     | successed times of tcp connections between current node and other nodes               |
| tcp-connectivity/${hostname/ip}_failed_count    | count     | failed times of tcp connections between current node and other nodes                  |
| tcp-connectivity/${hostname/ip}_success_rate    |           | success rate (successed/total) of tcp connection between current node and other nodes |
| tcp-connectivity/${hostname/ip}_time_min        | time (ms) | mininum latency of tcp connections between current node and other nodes               |
| tcp-connectivity/${hostname/ip}_time_max        | time (ms) | maximum latency of tcp connections between current node and other nodes               |
| tcp-connectivity/${hostname/ip}_time_avg        | time (ms) | average latency of tcp connections between current node and other nodes               |

### `gpcnet-network-test` / `gpcnet-network-load-test`

#### Introduction

Distributed test, test the global network performance and congestion,
performed by [GPCNET](https://github.com/netbench/GPCNET)

gpcnet-network-test: Full system network tests in random and natural ring, alltoall and allreduce, at least 2 nodes

gpcnet-network-load-test: Select full system network tests run with four congestors to measure network congestion or contention, at least 10 nodes

 - supporting network tests: RR Two-sided Lat (8 B), RR Get Lat (8 B), RR Two-sided BW (131072 B), RR Put BW (131072 B), RR Two-sided BW+Sync (131072 B), Nat Two-sided BW (131072 B), Multiple Allreduce (8 B), Multiple Alltoall (4096 B)
 - supporting congestors: Alltoall (4096 B), Two-sided Incast (4096 B), Put Incast (4096 B), Get Bcast (4096 B)

#### Metrics

| Metrics                                                 | Unit                   | Description                                                                                                                                                                |
|---------------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gpcnet-network-test/rr_two-sided_lat_${stat}            | time (us)              | statistical values(min, max, avg, 99%, 99.9%) obtained by all nodes use algorithm 'random ring communication pattern two-side latency' for network testing                 |
| gpcnet-network-test/rr_two-sided+sync_bw_${stat}        | bandwidth (MiB/s/rank) | fstatistical values(min, max, avg, 99%, 99.9%) obtained by all nodes use algorithm 'random ring communication pattern two-side bandwidth with barrier' for network testing |
| gpcnet-network-test/multiple_allreduce_time_${stat}     | time (us)              | statistical values(min, max, avg, 99%, 99.9%) obtained by all nodes use algorithm 'multiple allreduce bandwidth' for network testing                                       |
| gpcnet-network-test/rr_get_lat_${stat}                  | bandwidth (MiB/s/rank) | statistical values(min, max, avg, 99%, 99.9%) obtained by all nodes use algorithm 'RR GetLat (8 B)' for network testing                                                    |
| gpcnet-network-test/rr_two-sided_bw_${stat}             | bandwidth (MiB/s/rank) | statistical values(min, max, avg, 99%, 99.9%) obtained by all nodes use algorithm 'RR Two-sidedBW (131072 B)' for network testing                                          |
| gpcnet-network-test/nat_two-sided_bw_${stat}            | bandwidth (MiB/s/rank) | statistical values(min, max, avg, 99%, 99.9%) obtained by all nodes use algorithm 'Nat Two-sidedBW (131072 B)' for network testing                                         |
| gpcnet-network-test/multiple_alltoall_bw_${stat}        | bandwidth (MiB/s/rank) | statistical values(min, max, avg, 99%, 99.9%) obtained by all nodes use algorithm 'Multiple Alltoall (4096 B)' for network testing                                         |
| gpcnet-network-load-test/rr_two-sided_lat_x_${stat}     | factor (x)             | summary about congestion impact factor of the network test algorithm                                                                                                       |
| gpcnet-network-load-test/rr_two-sided+sync_bw_x_${stat} | factor (x)             | summary about congestion impact factor of the network test algorithm                                                                                                       |
| gpcnet-network-load-test/multiple_allreduce_x_${stat}   | factor (x)             | summary about congestion impact factor of the network test algorithm                                                                                                       |

### `ib-traffic`

#### Introduction

Measure the InfiniBand performance under multi nodes' traffic pattern.

The direction between client and server can be 'cpu-to-cpu'/'gpu-to-gpu'/'gpu-to-cpu'/'cpu-to-gpu'.

The traffic pattern is defined in a config file, which is pre-defined for one-to-many, many-to-one and all-to-all patterns.
Each row in the config is one round, and all pairs of nodes in a row run ib command simultaneously.

Besides the above three patterns, ib-traffic also supports topology-aware traffic pattern. To run ib-traffic with topology-aware
pattern, the user needs to specify 3 required (and 2 optional) parameters in YAML config file:
   - --pattern	&emsp;**topo-aware**
   - --ibstat	&emsp;**path to ibstat output**
   - --ibnetdiscover	&emsp;**path to ibnetdiscover output**
   - --min_dist	&emsp;**minimum distance of VM pairs (optional, default 2)**
   - --max_dist	&emsp;**maximum distance of VM pairs (optional, default 6)**

Each row in the config file has all VM pairs with a fixed distance (#hops). That's by default, 1st, 2nd, 3rd row has all VM pairs
with topology distance of 2, 4, 6, respectively.

#### Metrics

| Metrics                                                                                     | Unit             | Description                                                                                                                                                                                                                                                                                                                 |
|---------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ib-traffic/ib\_write\_bw\_${msg_size}\_${direction}\_${line}\_${pair}:${server}\_${client}  | bandwidth (GB/s) | The max bandwidth of perftest (ib_write_bw, ib_send_bw, ib_read_bw) using ${msg_size} with ${direction}('cpu-to-cpu'/'gpu-to-gpu'/'gpu-to-cpu'/'cpu-to-gpu') run between the ${pair}<sup>th</sup> node pair in the ${line}<sup>th</sup> line of the config, ${server} and ${client} are the hostname of server and client.  |
| ib-traffic/ib\_write\_lat\_${msg_size}\_${direction}\_${line}\_${pair}:${server}\_${client} | time (us)        | The max latency of perftest (ib_write_lat, ib_send_lat, ib_read_lat) using ${msg_size} with ${direction}('cpu-to-cpu'/'gpu-to-gpu'/'gpu-to-cpu'/'cpu-to-gpu') run between the ${pair}<sup>th</sup> node pair in the ${line}<sup>th</sup> line of the config, ${server} and ${client} are the hostname of server and client. |


## Computation-communication Benchmarks

### `computation-communication-overlap`

#### Introduction

Test the performance of single node when communication and computation overlap.

#### Metrics

| Name                                                  | Unit      | Description                                                  |
|-------------------------------------------------------|-----------|--------------------------------------------------------------|
| pytorch-computation-communication-overlap/mul_time    | time (ms) | Time of communication and mul kernel computation overlap.    |
| pytorch-computation-communication-overlap/matmul_time | time (ms) | Time of communication and matmul kernel computation overlap. |

####

### `sharding-matmul`

#### Introduction

Test the performance of large scale matmul operation with multiple GPUs:
* allreduce: Each GPU will calculate part of the MM calculation, and use AllReduce to merge all data into one tensor.
* allgather: Each GPU will calculate part of the MM calculation, and use AllGather + Concat to merge all data into one tensor.

#### Metrics

| Name                                   | Unit      | Description                              |
|----------------------------------------|-----------|------------------------------------------|
| pytorch-sharding-matmul/allreduce_time | time (ms) | Time of sharding matmul using allreduce. |
| pytorch-sharding-matmul/allgather_time | time (ms) | Time of sharding matmul using allgather. |

### `dist-inference`

#### Introduction

Test the performance of distributed model inference. Support both PyTorch implementation and cpp implementation.

#### Metrics

| Name                                            | Unit      | Description                                           |
|-------------------------------------------------|-----------|-------------------------------------------------------|
| pytorch-dist-inference/step_times               | time (ms) | Average time of model inference runs.                 |
| pytorch-dist-inference/step_times_${percentile} | time (ms) | Tail (50,90,95,99,99.9) time of model inference runs. |

## Storage Benchmarks

### `disk-benchmark`

#### Introduction

Measure the disk performance through [FIO](https://github.com/axboe/fio/tree/0313e938c9c8bb37d71dade239f1f5326677b079).

#### Metrics

| Name                                                          | Unit         | Description                                              |
|---------------------------------------------------------------|--------------|----------------------------------------------------------|
| disk-benchmark/${disk_name}_rand_read_write_bs                | size (bytes) | Disk random read write block size.                       |
| disk-benchmark/${disk_name}_rand_read_write_read_iops         | IOPS         | Disk random read write read IOPS.                        |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_95.0  | time (ns)    | Disk random read write read latency in 95.0 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_99.0  | time (ns)    | Disk random read write read latency in 99.0 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_read_lat_ns_99.9  | time (ns)    | Disk random read write read latency in 99.9 percentile.  |
| disk-benchmark/${disk_name}_rand_read_write_write_iops        | IOPS         | Disk random read write write IOPS.                       |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_95.0 | time (ns)    | Disk random read write write latency in 95.0 percentile. |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_99.0 | time (ns)    | Disk random read write write latency in 99.0 percentile. |
| disk-benchmark/${disk_name}_rand_read_write_write_lat_ns_99.9 | time (ns)    | Disk random read write write latency in 99.9 percentile. |
