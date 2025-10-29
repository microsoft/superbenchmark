[knowledge]

knowledge 1
Theoretical performance on both {target} and {baseline} design was compared using the specification value obtained from manufacturer. To produce a comprehensive performance validation on all basic building hardware component, GPU Computation,  GPU Memory, GPU communication, Networking, etc.

knowledge 2
Comparison between the {target} and {baseline} testbed was performed on the peak value of each metric, to understand how well the {target} design performs against the {baseline} testbed.
The comparison result is presented using the following metrics:
\begin{equation}
  PerformanceRatio = \frac{LatencyTimePerformance_{{baseline}}}{LatencyTimePerformance_{{target}}}
\end{equation}
For kernel launch and latency performance metrics. A value of \textgreater1 indicates {target} performs better.
\begin{equation}
  PerformanceRatio = \frac{BandwidthPerformance_{{target}}}{BandwidthPerformance_{{baseline}}}
\end{equation}
For FLOPS and Bandwidth performance metrics. A value of \textgreater1 indicates {target} performs better.

knowledge 3
Below is a complete list of the hardware component and benchmark metrics to validate this component.
The string betwee ** and ** is the name of hardware component, the string betwee * and * is the name of the benchmark metric.
**GPU Computation**: Anticipated the GPU GEMM FLOPS for different float or int data types, with Tensor Core or Matrix Core.
*FP8_TensorCore (GFLOPS)*: GEMM float8 peak FLOPS with Tensor Core (Matrix Core).
*FP6_TensorCore (GFLOPS)*: GEMM float6 peak FLOPS with Tensor Core (Matrix Core).
*FP4_TensorCore (GFLOPS)*: GEMM float4 peak FLOPS with Tensor Core (Matrix Core).

**GPU Memory**: Hardware specification for GPU memory component.
*Capacity (GB)*: Total GPU memory available to users. 
*Peak BW (GB)*: Peak READ + WRITE memory bandwidth.

**GPU Communication**: Anticipated memory copy bandwidth performed by GPU SM/DMA engine.
*GPU-to-GPU BW (GB/s)*: The unidirectional bandwidth of GPU Peer to peer communicaton.
*All-to-All BW (GB/s)*: The unidirectional bandwidth of all GPUs in the system are running all-to-all traffic.
*All-Reduce BW (GB/s)*: The projected All-Reduce bus bandwidth.

**Scale-out Networking** Anticipated total scale-out bandwidth.
*Total IB BW (GB/s)*: 8x InfiniBand loopback verbs bandwidth.

**Energy**:
*TBP (W)*: Total Board Power by design.
*Cooling*: The cooling method.
