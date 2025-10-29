[knowledge]

knowledge 1
Benchmarking on both {target} and {baseline} design was executed by SuperBench tool developed by Microsoft Research and Azure HPC/AI, to produce a comprehensive performance validation on all basic building hardware component, including CPU Memory, GPU Computation,  GPU Memory, GPU PCIe, GPU-to-GPU communication, IB NIC, etc.

knowledge 2
Comparison between the {target} and {baseline} testbed was performed on the peak value of each metric, to understand how well the {target} design performs against the {baseline} testbed.
The comparison result is presented using the following metrics:
\begin{equation}
  \mathit{PerformanceRatio} = \frac{\mathit{BandwidthPerformance}_{\mathit{{baseline}}}}{\mathit{BandwidthPerformance}_{\mathit{{target}}}}
\end{equation}
For FLOPS and Bandwidth performance metrics. A value of $<1$ indicates {target} performs better.

knowledge 3
Below is a complete list of the hardware component and benchmark metrics to validate this component.
The string betwee ** and ** is the name of hardware component, the string betwee * and * is the name of the benchmark metric.
**Computation**: 
*BF16/FP8/FP4 (TFLOPS/IOPS)*: GEMM peak FLOPS for bfloat16, float8, float4 precisions.

**Communication**: Measure the memory copy bandwidth performed by Device/GPU SM/DMA engine.
*Device to Host BW (GB/s)*: Measures the peak device to host copy bandwidth.
*Host to Device BW (GB/s)*: Measures the peak host to device copy bandwidth.
*Device to Device BW (GB/s)*: The unidirectional bandwidth of one device reading from or writing to another device's memory.
