
[specification]


[content of output]
/section
use the section title of 'Multi-Node - Software Library - Communication Performance (scale 2/4/8)'
Give a brief description of the how the test was performed.
Generate one paragraph to present the specific metric definition used to present the comparison results, in formula format

/subsection
briefly explain MSCCL

/subsection
In this subsection, focus on how much MSCCL can improve over RCCL on the BusBandwidth ratio that {target} can reach compared with {baseline}.
1.Read the performance tables of 'BusBandwidth tables, single node (NVSwitch SHARP disabled)' and 'Benchmark and Projected Performance Table' from case part.
2.Generate a paragraph talking about {target} largely underperforms the {baseline} on bus bandwidth when using RCCL library. Use an averaged bus bandwidth speedup value among message size range to demonstrate this conclusion. Use separated sentences for each communication algorithm (All-Reduce, All-to-All, All-Gather), and for each communication algorithm, use one sentence to describe the comparison results and findings for communication library RCCL. An average speedup close to 80\% means {target} is approaching {baseline}. An average speedup lower than to 70\% means {target} has large gap compared with {baseline}.
3.Generate a paragraph talking about {target} can have much better performance and approach {baseline} on bus bandwidth when using MSCCL library. Use an averaged bus bandwidth speedup value among message size range to demonstrate this conclusion. Use separated sentences for each communication algorithm (All-Reduce, All-to-All, All-Gather), and for each communication algorithm, use one sentence to describe the comparison results and findings for communication library MSCCL. An average speedup close to 80\% means {target} is approaching {baseline}. An average speedup lower than to 70\% means {target} has large gap compared with {baseline}.
4.Generate a paragraph talking about {target} can almost reach its projected performance on peak bus bandwidth, with the help of Microsoft technology - MSCCL. Use performance ratio, {target} of projected, to demonstrate this conclusion.
5.Generate a paragraph to analyze how {target} and {baseline} performs on distributed training workload by considering how they perform on the mostly used message sizes in distributed training workload.

/subsection
In this subsection, focus on how much MSCCL can improve over RCCL on the Latency ratio that {target} can reach compared with {baseline}.
1.Read the performance tables of 'Latency tables, single node (NVSwitch SHARP disabled)' from case part, use these rule to interpret the values from the table in this subsection: the performance ratio values in these tables should be interpreted in this way: a performance ratio value larger than 1 means {target} latency is smaller, meaning {target} outperforms {baseline}; a performance ratio value smaller than 1 means {target} latency is larger, meaning {target} underperforms {baseline}.
2.Generate a paragraph talking about {target} largely underperforms the {baseline} on latency when using RCCL library. Use an averaged latency speedup value among message size range to demonstrate this conclusion. Use separated sentences for each communication algorithm (All-Reduce, All-to-All, All-Gather), and for each communication algorithm, use one sentence to describe the comparison results and findings for communication library RCCL. An average speedup close to 80\% means {target} is approaching {baseline}. An average speedup lower than to 70\% means {target} has large gap compared with {baseline}.
3.Generate a paragraph talking about {target} can have much better performance and approach {baseline} on latency when using MSCCL library. Use an averaged latency speedup value among message size range to demonstrate this conclusion. Use separated sentences for each communication algorithm (All-Reduce, All-to-All, All-Gather), and for each communication algorithm, use one sentence to describe the comparison results and findings for communication library MSCCL. An average speedup close to 80\% means {target} is approaching {baseline}. An average speedup lower than to 70\% means {target} has large gap compared with {baseline}.
4.Generate a paragraph talking about {target} can almost reach its projected performance on latency, with the help of Microsoft technology - MSCCL. Use performance ratio, {target} of projected, to demonstrate this conclusion.
5.Generate a paragraph to analyze how {target} and {baseline} performs on distributed inference workload by considering how they perform on the mostly used message sizes in distributed inference workload.

\subsection
Benchmark Performance Table
use expression like "Please refer to Table \ref{tab:xxx} for detailed benchmark results" in this section, replace the xxx with the real label name presented in case field, while you must keep the 'tab' string to avoid build error.


[case]

