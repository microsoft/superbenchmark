[specification]


[content of the report]

The content of this section of report should include:
\section
use the section title of 'Typical Small Model Training Performance'
Give a brief description of the how the test was performed.
Generate one paragraph to present the specific metric definition used to present the comparison results, in formula format.

\subsection
Introduce SuperBench's function and capability, use SuperBench as the subsection title, this will help reader to understand the benchmark result after understanding the benchmark tool.

\subsection
Analyze the throughput speedup from {target} to {baseline}, for apple-to-apple. This will let the reader know how to compare {target} to {baseline} for their performance of serveing machine learning training workload. Do this in the following steps.
1.Read from the tables (superbench-1, superbench-2, superbench-3) in the case part whose title contains 'apple-to-apple', Generate an paragraph to highlighted the average throughput speedup of {target} to {baseline} for transformer models, CNN models, and LSTM model, for FP16. 
2.Read from the tables (superbench-1, superbench-2, superbench-3) in the case part whose title contains 'apple-to-apple', Generate an paragraph to highlighted the average throughput speedup of {target} to {baseline} for transformer models, CNN models, and LSTM model, for FP32.
3.Read from the tables (superbench-1, superbench-2, superbench-3) in the case part whose title contains 'apple-to-apple', Generate an paragraph to highlighted the average throughput speedup of {target} to {baseline} for transformer models, for FP8.

\subsection
Analyze the throughput speedup from {target} to {baseline}, for peak. This will let the reader know how to compare {target} to {baseline} for their performance of serveing machine learning training workload. Do this in the following steps.
1.Read from the tables (superbench-4, superbench-5, superbench-6) in the case part whose title contains 'peak', Generate an paragraph to highlighted the average throughput speedup of {target} to {baseline} for transformer models, CNN models, and LSTM models, for FP16. 
2.Read from the tables (superbench-4, superbench-5, superbench-6) in the case part whose title contains 'peak', Generate an paragraph to highlighted the average throughput speedup of {target} to {baseline} for transformer models, CNN models, and LSTM models, for FP32.
3.Read from the tables (superbench-4, superbench-5, superbench-6) in the case part whose title contains 'peak', Generate an paragraph to highlighted the average throughput speedup of {target} to {baseline} for transformer models, for FP8.

\subsection
In 'apple-to-apple' comparison, the batch size were not optimized for {target}, it uses the exact batch size as {baseline}.
In 'peak' comparison, the batch size were optimized for {target}
Compare the speedups between 'apple-to-apple' comparison and 'peak' comparison, make a conclusion with numbers to show the effect of the GPU memory capacity increasement.

\subsection
Benchmark results
use expression like "Please refer to Table \ref{tab:xxx} for detailed benchmark results" in this section, replace the xxx with the real label name presented in case field, while you must keep the 'tab' string to avoid build error.




[case]
