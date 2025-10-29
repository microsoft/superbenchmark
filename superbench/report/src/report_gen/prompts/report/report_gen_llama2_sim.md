[knowledge]

knowledge 1
The anticipated performance on Llama-2-70B inference was projected from previous generation SKUs, detailed description of the projecting method can be found in \ref{sec:efficiency}

knowledge 2

The comparison result is presented using the following metric:

  \begin{equation}
       \mathit{Speedup(CollectiveCommunicationLibrary)} =\frac{\mathit{StepLatency}_{\mathit{{baseline}}}}{\mathit{StepLatency}_{\mathit{{target}}}}
  \end{equation}
  To evaluate with the specified collective communication library. A value of \textgreater1 indicates {target} performs better.


knowledge 3
If a metric is throughput, larger value means better performance.
If a metric is latency, smaller value means better performance.


knowledge 4
Llama-2-70B workload description:
It helps evaluate how {target} and {baseline} perform to run Llama-2-70B inference service.
The workload measures the end-to-end latency of Llama-2-70B.

The Llama 2-70B model is a part of the Llama 2 family of large language models (LLMs) developed by Meta. These models are pretrained and fine-tuned generative text models, ranging in scale from 7 billion to 70 billion parameters. The Llama 2-70B model is specifically optimized for dialogue use cases and has been converted for the Hugging Face Transformers format.

The Llama 2-70B model uses an auto-regressive language model with an optimized transformer architecture. It employs supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. The model was trained on 2 trillion tokens of data from publicly available sources and does not include Meta user data.



