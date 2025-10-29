
[specification]




[content of the report]

The content of this section of report should include:
\section
use the title of "theoretical hardware peak performance (specification value)"

\subsection
Generate one paragraph to detailed explain how the expected specification value was got for {target} and {baseline}. They are provided to us by the GPU manufacturers.

\subsection
Summary
specifically to refer to the table \ref{tab:projected-1}
Generate one sentence to calculate and present the average value of "Ratio to Projected" by understanding the "Ratio to Projected" values of table "Benchmark and Projected Performance Table" in the case part.
Generate one sentence to conclude how well {target} performs against projected performance, using this rule to judge whether {target} achieved projected perofrmance by checking average value of "Ratio to Projected", if the average ratio is above 90%, it means it achieved projected performance, otherwise it means it failed to achieve projected performance.
Generate one sentence to list the benchmarks whose "Ratio to Projected" is below 90%, these components requires further optimization.
Generate one sentence to calculate and present the average value of "Ratio to Spec." by understanding the "Ratio to Spec." values of table "Benchmark and Projected Performance Table" in the case part.
Generate one sentence to present how much {target} performs to peak hardware performance defined in specification, using expression like:'the MI300x only achieve 66% of hardware performance defined in the specification'. If the average ratio is above 90%, it means it almost achieved peak hardware performance, otherwise it means there lies a great potential to further optimize for the peak performance.
specifically to refer to the table \ref{tab:projected-2}, Generate a sentence to calculate the ratio to spec average value for {baseline}, to conclude how much percentage {baseline} is achieving to its peak hardware performance defined in specification, and make suggestions by comparison the average ratio to spec between {target} and {baseline}.

\subsection
Comparison of spcification value on {target} and {baseline}
use expression like "Please refer to Table \ref{tab:xxx} for detailed benchmark results" in this section, replace the xxx with the real label name presented in case field, while you must keep the 'tab' string to avoid build error.

Read from the table in the 'case' part, for each component in (*GPU Computation*, *GPU Communication*, *GPU Memory*), generate an itemize list with highligheted ratio values to highligh how well {target} design compares to {baseline} design by comparing the average ratio.

You should use below expressions:
'- for <component 1>: we can expect {target} to {baseline} performance in <benchmark metric 1> with a ratio of n1, and <benchmark metric 2> with a ratio of n2, and ...; we can expect {target} to {baseline} performance in <benchmark metric 1> with a ratio of n1 and <benchmark metric 2> with a ratio of n2, and...; we can expect {target} to {baseline} performance in <benchmark metric 1> with a ratio of n1 and <benchmark metric 2> with a ratio of n2, and...'
'- for <component 2>: ...'

[case]

