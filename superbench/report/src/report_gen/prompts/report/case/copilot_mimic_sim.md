
[specification]


[content of the report]

The content of this section of report should include:
\section
use the section title of 'Anticipated CoPilot Mimic Inference Performance'
Give a brief description of the how the test was performed.
Generate one paragraph to present the specific metric definition used to present the comparison results, in formula format. If there are more than one formula, you need to include them all.

\subsection
CoPilot Mimic Inference
Introduce the details of the CoPilot Mimic inference workload structure used in this test, including how many layers, what precision is used, what metric is measured, and detailed configurations of each layer. this will help reader to understand the results.

\subsection
Results on CoPilot
In this subsection, focus on how is {target}'s CoPilot Mimic inference workload performance compared with {baseline} when using MSCCL library. Please do not ask user to refer to tables for details, you must summarize the results into one paragraph using following steps.
\subsubsection, FP16
1.Read the performance tables of 'Per-Step Latency (ms) and Speedup, CoPilot Mimic, FP16' from case part.
2.Generate a sentence summarizing the average speedup ratio {target} over {baseline} among full batch size range when using MSCCL library.
3.Generate a sentence summarizing the average speedup ratio on smallest batch size, and the average speedup ratio on largest batch size, and summarize if there is an obvious trend of ratio increase or decrease along with batch size increasement.
\subsubsection, FP8
1.Read the performance tables of 'Per-Step Latency (ms) and Speedup, CoPilot Mimic, FP8' from case part.
2.Generate a sentence summarizing the average speedup ratio {target} over {baseline} among full batch size range when using MSCCL library.
3.Generate a sentence summarizing the average speedup ratio on smallest batch size, and the average speedup ratio on largest batch size, and summarize if there is an obvious trend of ratio increase or decrease along with batch size increasement.
\subsubsection, FP4
1.Read the performance tables of 'Per-Step Latency (ms) and Speedup, CoPilot Mimic, FP4' from case part.
2.Generate a sentence summarizing the average speedup ratio {target} over {baseline} among full batch size range when using MSCCL library.
3.Generate a sentence summarizing the average speedup ratio on smallest batch size, and the average speedup ratio on largest batch size, and summarize if there is an obvious trend of ratio increase or decrease along with batch size increasement.

\subsection
Performance Table
use expression like "Please refer to Table \ref{tab:xxx} for detailed benchmark results" in this section, replace the xxx with the real label name presented in case field, while you must keep the 'tab' string to avoid build error.

[case]

