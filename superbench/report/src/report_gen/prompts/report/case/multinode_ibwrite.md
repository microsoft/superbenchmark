
[specification]


[content of output]
/section
use the section title of 'Multi-Node - InfiniBand - Communication Performance (scale 2/4/8)'
Give a brief description of the how the test was performed.
Generate one paragraph to present the specific metric definition used to present the comparison results, in formula format


/subsection
IB Write Bandwidth
In this subsection, focus on how well {target} performs over {baseline}.
1.Read the performance tables of 'Multi-Node: IB Write Bandwidth (GB/s)' from case part.
2.Generate a paragraph to compare how {target} performs over {baseline} for CPU to CPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.
3.Generate a paragraph to compare how {target} performs over {baseline} for CPU to GPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.
4.Generate a paragraph to compare how {target} performs over {baseline} for GPU to CPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.
5.Generate a paragraph to compare how {target} performs over {baseline} for GPU to GPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.

/subsection
IB Write Latency
In this subsection, focus on how well {target} performs over {baseline}.
1.Read the performance tables of 'Multi-Node: IB Write Latency (GB/s)' from case part.
2.Generate a paragraph to compare how {target} performs over {baseline} for CPU to CPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.
3.Generate a paragraph to compare how {target} performs over {baseline} for CPU to GPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.
4.Generate a paragraph to compare how {target} performs over {baseline} for GPU to CPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.
5.Generate a paragraph to compare how {target} performs over {baseline} for GPU to GPU traffic, show the average ratio, show the ratio range, show the message size which produce the maximum ratio.

\subsection
Benchmark Performance Table
use expression like "Please refer to Table \ref{tab:xxx} for detailed benchmark results" in this section, replace the xxx with the real label name presented in case field, while you must keep the 'tab' string to avoid build error.


[case]

