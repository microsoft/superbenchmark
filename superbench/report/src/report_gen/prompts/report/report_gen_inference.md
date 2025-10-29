[knowledge]


knowledge 1
If a metric is throughput, larger value means better performance.
If a metric is latency, smaller value means better performance.


knowledge 2
CoPilot Mimic workload description:
CoPilot Mimic mimics the workload of real CoPilot inference service. It helps evaluate how {target} and {baseline} perform to run CoPilot inference service.
The workload measures the end-to-end latency of a 50-layer mimic model with FP16 precision. Each layer contains two GEMM operations and one all-reduce operation, simulating the sharded feed-forward module of transformer block in CoPilot model.
The first GEMM operation is B[m, k] * A[k, n] + C[m, n] => D[m, n].
The second GEMM operation is E[k, m] * D[m, n] + F[k, n] = G[k, n].
The all-reduce operation does in-place all-reduce for G[k, n].
n is batch size from 1 to 2048. m and k have different combinations, including (2208, 5608), (11216, 320), (1536, 4608), (2208, 4608), (9216, 320), (9216, 768), (9216, 430), (32, 4608), to simulate different workloads. All-reduce size is for k*n elements, from 860 bytes to 11MB.
The workload is tested on 1 node for both {target} and {baseline}.

knowledge 3
GPT3-175B Mimic workload description:
GPT3-175B Mimic mimics the workload of real GPT3-175B inference service. It helps evaluate how {target} and {baseline} perform to run GPT3-175B inference service.
The workload measures the end-to-end latency of a 50-layer mimic model with FP16 precision. Each layer contains two GEMM operations and one all-reduce operation, simulating the sharded feed-forward module of transformer block in GPT3-175B model.
The first GEMM operation is B[m, k] * A[k, n] + C[m, n] => D[m, n].
The second GEMM operation is E[k, m] * D[m, n] + F[k, n] = G[k, n].
The all-reduce operation does in-place all-reduce for G[k, n].
n is batch size from 1 to 2048. m is 6144 and k is 12288 to simulate GPT3-175B workload. All-reduce size is for k*n elements, from 12KB to 24MB.
The workload is tested on 1 node for both {target} and {baseline}.

knoeledge 4

The comparison result is presented using the following metric:

  \begin{equation}
       \mathit{Speedup(NCCL/RCCL/MSCCL)} =\frac{\mathit{LayerLatencyNCCL/RCCL/MSCCL}_{\mathit{{baseline}}}}{\mathit{LayerLatencyNCCL/RCCL/MSCCL}_{\mathit{{target}}}}
  \end{equation}
  To evaluate with the specified collective communication library. A value of \textgreater1 indicates {target} performs better.
