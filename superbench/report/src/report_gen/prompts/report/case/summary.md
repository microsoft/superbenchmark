your task is to summarize the key findings, conclusion and summary with highlighed numbers about performance and resource utilization for each of the below benchmark comparison between {target} and {baseline}, and {target}'s benchmark results to its expected results.

knowledge you can use:
When you compare the performance or resource utilization, always use a 0.5X or 1.6X value to present a quantitive comparison.
When you present a number, use blue color bold font for highlight, e.g., \textbf{\textcolor{blue}{1.00X}}.
When you generate the content, you should keep the existing bold font to make the content focused.
A ratio higher than 0.8 means {target} achieved expected performance
A ratio lower than 0.8 means {target} did not achieve expected performance
A ratio higher than 1 means {target} outperforms {baseline}
A ratio between 0.95 and 1 means {target} performs similar to {baseline}
A ratio lower than 0.95 means {target} underperform {baseline}

[content of output]:
\section{Executive Summary}
You should generate the output using {format} format, take the expressions in 'draft' part as example, rephrase the draft for better readability and clarity, generate the highlights using the results from 'case' part.

Generate one sentence to present the average ratio and conclusions on how \textbf{Performance against Expectation}, evaluate whether {target} achieved peak performance defined in specification. Generate one sentence to summarize the components and ratio values for which {target} achived expectation, using the expression like 'To be specific, the GB200 achieved approximately \textbf{\textcolor{blue}{x\%}} of the theoretical GPU-GPU bandwidth, reaching around \textbf{\textcolor{blue}{n GB/s}}'. Generate one sentence to summarize those components and ratio values for which {target} did not achieve expectation, using the expression like 'the GB200 achieved only \textbf{\textcolor{blue}{x\%}} of the theoretical GEMM throughput with \textbf{\textcolor{blue}{n TFLOPs}}'

Generate one sentence to present the summary on how {target} performs agains {baseline} on \textbf{Raw Hardware Performance}, Generate one sentence to summarize the components and ratios in which {target} outperforms {baseline}, and those components and ratios in which {target} underperforms {baseline}.

Generate one sentence to present a description on how \textbf{Software Library} are benchmarked and evaluted, the reference content is in the 'draft' part.

Generate one sentence to present a description on the software library availability status.

Generate one sentence to present the key conclusions about Software Library Performance. Describe the software library performance status. For the \textbf{computation library (cuBLAS/cuBLASLt)} performance, show the peak comparison ratio on fp16 of {target} to {baseline} from the raw hardware performance section, then show the comparison ratio for math-limited kernels and memory-limited kernels from the computation library part, and conclude what should be optimized considering the different kernel types. For the \textbf{communication library (NCCL)} performance, please present the key findings on how {target} performs against {baseline}, by showing the average ratio of BusBW for AllReduce, AlltoAll and AllGather.

Generate one sentence to present a description on the software library reliability status.

Generate one sentence to present the key conclusions about \textbf{End-to-end model Performance}. In terms of small model performance, show the comparison ratio range of {target} over {baseline} for each precision respectively.

Generate an itemized list to propose key executable suggestions on further benchmarking and performance optimization.

[draft]
