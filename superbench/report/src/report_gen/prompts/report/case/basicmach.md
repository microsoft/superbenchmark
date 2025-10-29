
[specification]



[content of the report]

The content of this section of report should include:
\section
use the section title of 'Raw Hardware Performance'
Generate one paragraph to desscribe in details what benchmark was executed, how the metrics were compared.
Generate one paragraph to present the specific metric definition used to present the comparison results, in formula format 

\subsection
Benchmark Metrics
Generate an itemized list to describe in details of the definition of the benchmark metrics for each hardware component. You need to include the details of the matrix shape m n k values in numbers, communication directions, message sizes in numbers, implementations.

\subsection
Benchmark Summary
Generate one sentence to calculate and present the average value of "Perf. Ratio" of table "Raw Hardware Performance Comparison" in the case part, for each of the below components: GPU Computation(FP16, BF16, FP8, INT8), GPU Kernel Launch, GPU Memory, GPU PCIe, and GPU Communication, this mean {target} is producing a x percentage performance to {baseline}.
Generate one paragraph by comparing the above calculated average performance ratio values, and conclude how well {target} performs against {baseline}. Please use this rule to judge whether {target} outperforms or underperforms {baseline} by checking the whether the average ratio value is higher than 90% or not. Do these only on these selected benchmarks: GPU Computation(FP16, BF16, FP8, INT8), GPU Kernel Launch, GPU Memory, GPU PCIe, and GPU Communication.



\subsection
Benchmark Performance Table
use expression like "Please refer to Table \ref{tab:xxx} for detailed benchmark results" in this section, replace the xxx with the real label name presented in case field, while you must keep the 'tab' string to avoid build error.

for GPU Computation, generate one sentence with highligheted ratio values to describe the comparison result between {target} to {baseline} for each benchmark metric separately. if the performance ratio is larger than 1.1, it means {target} outperforms {baseline}; if the performance ratio is smaller than 0.90, it means {target} underperforms {baseline}; otherwise they perform similarly.

for GPU PCIe, generate one sentence with highligheted ratio values to describe the comparison result between {target} to {baseline} for each benchmark metric separately. if the performance ratio is larger than 1.1, it means {target} outperforms {baseline}; if the performance ratio is smaller than 0.90, it means {target} underperforms {baseline}; otherwise they perform similarly.

for GPU Communication, generate one sentence with highligheted ratio values to describe the comparison result between {target} to {baseline} for each benchmark metric separately. if the performance ratio is larger than 1.1, it means {target} outperforms {baseline}; if the performance ratio is smaller than 0.90, it means {target} underperforms {baseline}; otherwise they perform similarly.

for GPU Memory, generate one sentence with highligheted ratio values to describe the comparison result between {target} to {baseline} for each benchmark metric separately. if the performance ratio is larger than 1.1, it means {target} outperforms {baseline}; if the performance ratio is smaller than 0.90, it means {target} underperforms {baseline}; otherwise they perform similarly.

for CPU Memory, generate one sentence with highligheted ratio values to describe the comparison result between {target} to {baseline} for each benchmark metric separately. if the performance ratio is larger than 1.1, it means {target} outperforms {baseline}; if the performance ratio is smaller than 0.90, it means {target} underperforms {baseline}; otherwise they perform similarly.

Use expression template: 
'for <component>, {target} outperforms {baseline} in <benchmark metric 1> with a ratio of n1, and <benchmark metric 2> with a ratio of n2, and ...'
'for <component>, {target} performs very close to {baseline} in <benchmark metric 1> and <benchmark metric 2> with a ratio of n1-n2'

Generate a paragraph to categorize the <benchmark metrics> for which {target} outperforms {baseline}, Please list all benchmark metrics names in this category.
Generate a paragraph to categorize the <benchmark metrics> for which {target} underperforms {baseline}, Please list all benchmark metrics names in this category, instead of using 'others', use expression template:
'{target} underperforms {baseline} in <benchmark metric 1>, <benchmark metric 2> ...'
[case]

