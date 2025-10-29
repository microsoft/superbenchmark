[knowledge]

knowledge 1
The anticipated performance on LLaMA3-80B inference was projected from previous generation SKUs, detailed description of the projecting method can be found in \ref{sec:efficiency}


knoeledge 2

The comparison result is presented using the following metric:

  \begin{equation}
       \mathit{Speedup(CollectiveCommunicationLibrary)} =\frac{\mathit{StepLatency}_{\mathit{{baseline}}}}{\mathit{StepLatency}_{\mathit{{target}}}}
  \end{equation}
  To evaluate with the specified collective communication library. A value of \textgreater1 indicates {target} performs better.


knowledge 3
If a metric is throughput, larger value means better performance.
If a metric is latency, smaller value means better performance.

knowledge 4
LLaMA3-80B workload description:
Description
LLaMA 3 is the latest advancement in the LLaMA model family, open sourced by Meta.  Trained on a dataset with over 15T tokens, LLaMA 3 has demonstrated state-of-the-art performance across diverse benchmarks, showcasing enhanced capabilities in reasoning and code generation. Distinguishing itself from its predecessor, LLaMA 2, LLaMA 3 can now accommodate a longer input length (i.e. 8192) alongside with a vocabulary consisting of 128K tokens. Besides, LLaMA 3 adopts Grouped query attention to improve inference efficiency. In this report, we assess the inference efficiency of 70B LLaMA 3 on MI375X.
Configuration
Tensor parallelism is applied for LLaMA 3 inference. We evaluate both prefilling (i.e. input length > 1) and decoding phases (i.e. input length = 1). The batch size spans from 1 to the maximum capacity supported by vLLM without involving any batch scheduling.


knowledge 5
The workload is simulated on 1 node for both {target} and {baseline}.

