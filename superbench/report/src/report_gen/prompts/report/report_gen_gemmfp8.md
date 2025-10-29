[knowledge]

knowledge 1
SuperBench, developed by Microsoft Research and Azure HPC/AI, is an open-sourced standard tool for machine learning system performance validation.

knowledge 2
speedup of a to b is defined as perf(b)/perf(a), if a and b are throughput metrics. A speedup value larger than 1 indicates the b out performs a. A speedup value smaller than 1 indicates the b is slower than 1.

knowledge 3
The Roofline model is an intuitive visual performance model used to provide performance estimates of a given compute kernel or application running on multi-core, many-core, or accelerator processor architectures, by showing inherent hardware limitations, and potential benefit and priority of optimizations. By combining locality, bandwidth, and different parallelization paradigms into a single performance figure, the model can be an effective alternative to assess the quality of attained performance instead of using simple percent-of-peak estimates, as it provides insights on both the implementation and inherent performance limitations.

Here are the steps to apply the Roofline model:
- Identify the System Peaks: This includes peak computational performance (in FLOPs/sec) and peak memory bandwidth (in bytes/sec). 
- Draw the Roofline: The Roofline model is a plot with the arithmetic intensity (FLOPs/byte) on the x-axis and performance (FLOPs/sec) on the y-axis. Draw two lines: One line starts at the origin and has a slope equal to the peak memory bandwidth. This represents the memory-bound region where performance is limited by memory bandwidth. The other line is a horizontal line at the height of the peak computational performance. This represents the compute-bound region where performance is limited by the computational power of the hardware.
- Characterize Kernel: GEMM kernel performance could possibly limited by math or memory. To determine this, we often compare the kernel's arithmetic intensity to the ops:byte ratio of the GPU. An kernel is math limited on a given processor if the  arithmetic intensity is higher than the processor’s ops:byte ratio. Conversely, an it is memory limited if its arithmetic intensity is lower than the processor’s ops:byte ratio. 
Calculation formular of a kernel's arithmetic intensity: 
\begin{equation}
    ArithmeticIntensity = \frac{m \times n \times k}{m \times n + n \times k + m \times k}
\end{equation}
where m, n, k are the matrix shape.
Calculation formular of ops:byte ratio:
\begin{equation}
    Ratio_{Ops:Byte} = \frac{PeakFLOPs}{PeakMemoryBandwidth}
\end{equation}
the Ops:Byte ratio need to calculated for each device and precision.
- Plot Your Kernel: Plot the arithmetic intensity and performance of your kernel on the Roofline graph.
- Interpret the Results: If your kernel is below the roofline, it's not achieving peak performance. If it's in the memory-bound region, we can improve performance by reducing memory use. If it's in the compute-bound region, we can improve performance by enhancing computational efficiency.




knowledge 4
The theoretical FLOPs number for a math limited kernel on a GPU under a precision is the Peak FLOPs.
The theoretical FLOPs number for a memory limited kernel on a GPU under a precision is the product of arithmetic intensity and the Peak memory bandwidth.

knowledge 5
we run the GEMM TFLOPs test with different input matrix size of m, n ,k values under different precisions on {target} and {baseline} testbed using SuperBench, and then performed the Roofline Analysis on the acquired FLOPs data. The GEMM data points for analysis are 496, with a diverse range of m, n, k values [16, 16384] to cover from less intensive to more computationally demanding kernels.


The comparison result is presented as AverageSpeedup, defined as follows:
\begin{equation}
    AverageSpeedup = \frac{1}{n} \sum \frac{FLOPS_{{target}}}{FLOPS_{{baseline}}}
\end{equation}
The average operation is performed on math-limited kernels or memory-limit kernels. A value of \textgreater1 indicates {target} performs better.