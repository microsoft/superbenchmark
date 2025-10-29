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
**GPU Computation**: 
*Kernel Launch Event Time (ms)*: Launch latency measured in GPU time.
*Kernel Launch Wall Time (ms)*: Launch latency measured in CPU time.
*BF16/FP16/TF32/FP32/FP8/INT8 TC/MC (TFLOPS/IOPS)*: GEMM peak FLOPS for bfloat16, float16, float32, tensorfloat32, float8, float64 precisions, and GEMM peak IOPS for int8 precision, with Tensor Core or XDLOPS. Please refer to Table \ref{tab:tensorshape} for detailed matrix shaped used to produce the peak FLOPS.

**GPU Device and Host Communication**: Measure the memory copy bandwidth across Device and Host, this could be PCIe or other interconnection links, performed by NVIDIA or AMD bandwidth test tool.
*Device to Host (GB/s)*: Measures the peak device to host copy bandwidth.
*Host to Device (GB/s)*: Measures the peak host to device copy bandwidth.
*Local Host*: Meaning both device and host are in the same chip for CPU-GPU integrated architecture, or both device and host are under the same root complext for CPU-GPU discrete architecture.
*Remote Host*: Meaning the device and host are in the different chip for CPU-GPU integrated architecture, or the device and host are under different root complext for CPU-GPU discrete architecture.

**GPU Communication (GPU Comm.)**: Measure the memory copy bandwidth performed by GPU SM/DMA engine.
*GPU-to-GPU BW by SM/DMA Engine (GB/s)*: The unidirectional bandwidth of one GPU reading or writing self's memory using DMA engine or GPU SM.
*GPU-to-GPU all-to-all BW by SM (GB/s)*: The unidirectional bandwidth of one GPU when all GPUs in the system are running all-to-all traffic.
*NCCL/RCCL with IB BusBW (GB/s)*: NCCL/RCCL operation bus bandwidth with given message size of 16GB, crossing InfiniBand.
*NCCL/RCCL BusBW (GB/s)*: NCCL/RCCL operation bus bandwidth with given message size of 16GB.
*IB (GB/s)*: Measure the InfiniBand loopback verbs bandwidth, performed by OFED performance tests.

**GPU Memory**:
*READ WRITE BW*: Measures peak READ + WRITE memory bandwidth achievable when moving data across full GPU memory address space using GPU SM, implemented by Superbench gpu-copy-bw benchmark.

**CPU Memory**: Measure the memory copy bandwidth and latency across different CPU NUMA nodes. performed by Intel MLC Tool.
*Bandwidth NUMA A B (MB/s)*: A NUMA to B NUMA memory bandwidth.
*Latency NUMA A B (ns)*: A NUMA to B NUMA memory latency.
