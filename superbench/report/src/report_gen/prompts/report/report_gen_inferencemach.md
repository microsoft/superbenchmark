[knowledge]


knowledge 1
If a metric is throughput, larger value means better performance.
If a metric is latency, smaller value means better performance.


knowledge 2
Llama 3-8B description:
The LLAMA 3-8B model, released in 2024, is the latest iteration of state-of-the-art open-source large language model developed by Meta. The 8B variant of the LLAMA 3 model is equipped with 8 billion parameters, demonstrating exceptional performance across a wide range of industry benchmarks. It features a transformer-based architecture with approximately 32 layers, a hidden size of 4096, 32 attention heads, and 128256 vocabulary size. The LLAMA 3 stands out due to its increased vocabulary size, from 32k in LLAMA2 to 128k in LLAMA3.

Llama2-13B description:
The LLAMA 2-13B model is a variant of the LLAMA 2 series, a cutting-edge large language model developed by Meta. This model is built on the standard Transformer architecture, equipped with the 13 billion parameter, enhancing its capacity to capture intricate patterns in language. It consists of 40 layers, a hidden size of 5120, 40 attention heads and 32000 vocabulary size. This larger configuration allows for superior handling of complex and nuanced language tasks, benefiting from extensive pre-training on diverse datasets. The 13B variant stands out due to its increased context length, extending from 2048 to 4096.

Mistral 7B description:
The MISTRAL-7B model is a groundbreaking development in the field of computer science, specifically in the realm of language models. Developed by the innovative MISTRAL AI team, this model boasts an impressive 7.3 billion parameters, making it the most powerful language model of its size to date. It features a transformer-based architecture with 32 layers, a hidden size of 4096, 32 attention heads and 32768 vocabulary size. 

knowledge 3
description of the metrics
To comprehensively evaluate the performance of model inference, we have designed a set of metrics that capture Prompt Phase and Token Phase across various input and output configurations.
These metrics provide a comprehensive view of the model's performance across different use cases, from single-user scenarios to high-throughput, multi-user environments. By analyzing these metrics, we can identify the efficiency and scalability of the system under different usage scenarios., ensuring they meet the performance requirements of various applications.
 
\textbf{Prompt Phase}: This phase refers to the initial processing of the input sequence by the model. During the prompt phase, the model reads and understands the provided input text (prompt), setting up the necessary context for generating subsequent tokens. The latency and throughput during this phase are critical for determining how quickly the model can start generating meaningful responses.
 
\textbf{Token Phase}: This phase involves the sequential generation of tokens after the initial prompt has been processed. In the token generation phase, the model generates one token at a time based on the context provided by the prompt and previously generated tokens. The performance during this phase is measured by the speed and efficiency of generating each new token, which is crucial for applications requiring long text generation.

\subsubsection{Prompt Phase Metrics}

\begin{itemize}
    \item Time to first token: Measures the latency (in milliseconds) to generate the first output token given an input sequence during the prompt phase. This metric is crucial for evaluating the responsiveness of the model in generating initial predictions. A lower Time To First Token indicates a more efficient system.
    \item Max throughput: Measures the maximum number of tokens generated per second during the prompt phase when the batch size is greater than 1. This metric evaluates the model's throughput when processing multiple inputs simultaneously. A higher Max Throughput indicates a more efficient system.
\end{itemize}

\subsubsection{Prompt Phase and Token Phase Metrics}

\begin{itemize}
    \item Single user throughput: Measures the throughput (tokens per second) for generating sequences when there is only a single user, covering both the prompt and token generation phases. This metric evaluates the combined performance of the prompt and token generation phases. A higher Single User Max Throughput indicates a more efficient system.
    \item Multi user total Max throughput: Measures the maximum throughput (tokens per second) during the prompt and token generation phases when the batch size is greater than 1. This metric assesses the maximum processing speed in scenarios where multiple users are generating sequences simultaneously, summing up all tokens processed for all users in a second. A higher Multi User Max Throughput indicates a more efficient system.
\end{itemize}

knowledge 4

The comparison result is presented using the following metrics:

  \begin{equation}
       \mathit{Ratio_Latency} =\frac{\mathit{Latency}_{\mathit{{target}}}}{\mathit{Latency}_{\mathit{{baseline}}}}
  \end{equation}
  To evaluate the latency performance metric. A value of $<1$ indicates {target} performs better.


  \begin{equation}
       \mathit{Ratio_Throughput} =\frac{\mathit{Throughput}_{\mathit{{baseline}}}}{\mathit{Throughput}_{\mathit{{target}}}}
  \end{equation}
  To evaluate the throughput performance metric. A value of $<1$ indicates {target} performs better.