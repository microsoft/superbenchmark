[knowledge]

knowledge 1
Benchmarking on both {target} and {baseline} design was executed by SuperBench tool developed by Microsoft Research and Azure HPC/AI, to produce a comprehensive performance validation on all basic building hardware component, including CPU Memory, GPU Computation,  GPU Memory, GPU PCIe, GPU-to-GPU communication, IB NIC, etc.

knowledge 2
Comparison between the {target} and {baseline} testbed was performed on the peak value of each metric, to understand how well the {target} design performs against the {baseline} testbed.
The comparison result is presented using the following metrics:
\begin{equation}
  \mathit{PerformanceRatio} = \frac{\mathit{LatencyTimePerformance}_{\mathit{{baseline}}}}{\mathit{LatencyTimePerformance}_{\mathit{{target}}}}
\end{equation}
For kernel launch and latency performance metrics. A value of \textgreater1 indicates {target} performs better.
\begin{equation}
  \mathit{PerformanceRatio} = \frac{\mathit{BandwidthPerformance}_{\mathit{{target}}}}{\mathit{BandwidthPerformance}_{\mathit{{baseline}}}}
\end{equation}
For FLOPS and Bandwidth performance metrics. A value of \textgreater1 indicates {target} performs better.

knowledge 3
Below is a complete list of the hardware component and benchmark metrics to validate this component.
The string betwee ** and ** is the name of hardware component, the string betwee * and * is the name of the benchmark metric.
**GPU Computation**: (1) Measure GPU kernel launch latency, which is defined as the time range from the beginning of the launch API call to the beginning of the kernel execution. (2) Measure the GPU GEMM FLOPS for different float and int data types, with Tensor Core (XDLOPS), performed by NVIDIA [cutlass] or AMD [rocblas-bench].
*Kernel_Launch_Event_Time (ms)*: Launch latency measured in GPU time.
*BF16_TensorCore (GFLOPS)*: GEMM bfloat16 peak FLOPS with Tensor Core (XDLOPS), using matrix shape m=8192, n=8192, k=8192.
*FP16_TensorCore (GFLOPS)*: GEMM float16 peak FLOPS with Tensor Core (XDLOPS), using matrix shape m=16384, n=16384, k=8192.
*FP32_TensorCore (GFLOPS)*: GEMM float32 peak FLOPS with Tensor Core (XDLOPS), using matrix shape m=8192, n=8192, k=8192.
*FP64_TensorCore (GFLOPS)*: GEMM float64 peak FLOPS with Tensor Core (XDLOPS), using matrix shape m=8192, n=8192, k=8192.
*INT8_TensorCore (GIOPS)*: GEMM int8 peak IOPS with Tensor Core (XDLOPS), using matrix shape m=16384, n=16384, k=8192.

**GPU PCIe**: Measure the memory copy bandwidth across PCI-e, erformed by NVIDIA or AMD bandwidth test tool.
*Device_to_Host BW (GB/s)*: Measures the peak device to host copy bandwidth.
*Host_to_Device BW (GB/s)*: Measures the peak host to device copy bandwidth.

**GPU Communication**: Measure the memory copy bandwidth performed by GPU SM/DMA engine.
*GPU-to-GPU BW by SM/DMA Engine (GB/s)*: The unidirectional bandwidth of one GPU reading or writing self's memory using DMA engine or GPU SM.
*GPU-to-GPU all-to-all BW by SM (GB/s)*: The unidirectional bandwidth of one GPU when all GPUs in the system are running all-to-all traffic.
*NCCL/RCCL_with_IB BusBW (GB/s)*: NCCL/RCCL operation bus bandwidth with given message size of 16GB, crossing InfiniBand.
*NCCL/RCCL BusBW (GB/s)*: NCCL/RCCL operation bus bandwidth with given message size of 16GB.
*IB (GB/s)*: Measure the InfiniBand loopback verbs bandwidth, performed by OFED performance tests.

**GPU Memory**:
*READ_WRITE_BW*: Measures peak READ + WRITE memory bandwidth achievable when moving data across full GPU memory address space using GPU SM, implemented by Superbench gpu-copy-bw benchmark.

**CPU Memory**: Measure the memory copy bandwidth and latency across different CPU NUMA nodes. performed by Intel MLC Tool.
*Bandwidth_NUMA_A_B (MB/s)*: A NUMA to B NUMA memory bandwidth.
*Latency_NUMA_A_B (us)*: A NUMA to B NUMA memory latency.
