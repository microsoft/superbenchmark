[knowledge]

knowledge 1
For distributed model training, MB to GB message sizes of all-reduce, all-gather and all-to-all are the focus. Specifically, all-reduce is used for data parallelism, all-gather is used for tensor parallelism, and all-to-all is used for MoE parallelism.
For distributed model inference, KB to MB message sizes of all-reduce and all-gather are the focus. Both all-reduce and all-gather are used for tensor sharding.

knowledge 2
MSCCL
Microsoft Collective Communication Library (MSCCL) is a platform to execute custom collective communication algorithms for multiple accelerators supported by Microsoft Azure, developed with collaboration effort from Microsoft Research and Azure HPC/AI.
MSCCL is an inter-accelerator communication framework that is built on top of NCCL and uses its building blocks to execute custom-written collective communication algorithms. MSCCL vision is to provide a unified, efficient, and scalable framework for executing collective communication algorithms across multiple accelerators. To achieve this, MSCCL has multiple capabilities:
Programmibility: Inter-connection among accelerators have different latencies and bandwidths. Therefore, a generic collective communication algorithm does not necessarily well for all topologies and buffer sizes. MSCCL allows a user to write a hyper-optimized collective communication algorithm for a given topology and a buffer size. This is possbile through two main components: MSCCL toolkit and MSCCL runtime (this repo). MSCCL toolkit contains a high-level DSL (MSCCLang) and a compiler which generate an IR for the MSCCL runtime (this repo) to run on the backend. MSCCL will always automatically fall back to a NCCL's generic algorithm in case there is no custom algorithm. Example provides some instances on how MSCCL toolkit with the runtime works. Please refer to MSCCL toolkit for more information.
Profiling: MSCCL has a profiling tool NPKit which provides detailed timeline for each primitive send and receive operation to understand the bottlenecks in a given collective communication algorithms.

knowledge 3
Below is the description of the workload.
We run collective communication performance of MI300X and NDv5 SKUs.
The library we used are RCCL and MSCCL for MI300X.
The library we used are NCCL and MSCCL for NDv5.
The communication algorithm we run are all-reduce, all-gather and all-to-all.
The communication performance metric we collected are latency and busbw.
The scale we run covers 2 nodes.

knowledge 4
The comparison results are presented using the following metrics:

\begin{equation}
    \mathit{Speedup} = \frac{\mathit{BusBW}_{\mathit{{target}}}}{\mathit{BusBW}_{\mathit{{baseline}}}}
\end{equation}
For Bus Bandwidth metrics. A speedup value of \textgreater1 indicates {target} performs better.

\begin{equation}
    \mathit{PerformanceRatio} = \frac{\mathit{Latency}_{\mathit{{baseline}}}}{\mathit{Latency}_{\mathit{{target}}}}
\end{equation}
For Latency metrics. A performance ratio value of \textgreater1 indicates {target} performs better.