[knowledge]

knowledge 1
Large language models (LLMs) have become pivotal in natural language understanding and generation tasks.
This GPT3 training experiment aims to shed light on the efficiency and performance of the LLM when trained on {target} and {baseline} platforms.
The experiment utilizing the GPT3 6.7B model, a decoder-only Transformer architecture. This architecture aligns with recent advancements in the field, showcased by models like PaLM, OPT, and LLaMA. Our GPT 6.7B model boasts 6.7 billion parameters, a dimension of 4096, 32 attention heads, and 32 layers.
The training settings include data and sequence parallelism, FP16 precision, learning rate of 1.2e-4.
The experiments on {target} and {baseline} are both conducted with the same setting using the Megatron-DeepSpeed library, a comprehensive library optimized for training large language models. Megatron-DeepSpeed integrates elements from NVIDIA's Metatron-LM and DeepSpeed, providing comprehensive support for both NVIDIA and ROCm platforms. This choice ensured efficient and highly optimized training runs on the hardware platforms.

knowledge 2
We train a GPT 6.7B model on {target} and {baseline} VMs.

The comparison results are presented using the following metrics:
\begin{equation}
    \mathit{ThroughputSpeedup} = \frac{\mathit{TrainingThroughput}_{\mathit{{target}}}}{\mathit{TrainingThroughput}_{\mathit{{baseline}}}}
\end{equation}
For performance metrics. A value of \textgreater1 indicates {target} performs better.

\begin{equation}
  \mathit{MemoryUsageEfficiency} = \frac{\mathit{MemoryUsage}_{\mathit{{baseline}}}}{\mathit{MemoryUsage}_{\mathit{{target}}}}
\end{equation}
For Memory utilization metrics. A value of \textgreater1 indicates {target} uses memory resources more efficiently.

\begin{equation}
  \mathit{FLOPsUtlization} = \frac{\mathit{FLOPSUsage}_{\mathit{{target}}}}{\mathit{FLOPSUsage}_{\mathit{{baseline}}}}
\end{equation}
For Memory utilization metrics. A value of \textgreater1 indicates {target} uses FLOPS resources more efficiently.


knowledge 3
OOM in a GPU memory ratio means out of memory
