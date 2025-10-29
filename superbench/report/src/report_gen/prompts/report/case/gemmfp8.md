
[specification]


[content of the report]

\section
use the section title of 'Software Library - GEMM Performance (FP8)'
Give a detailed description of the how this benchmark was performed, including how many data points collected, what shapes of kernel are tested. no need to add a subtitle.
Generate one paragraph to present the specific metric definition used to present the comparison results, in formula format 
\subsection
This subsection's goal is to present the analysis model or algorithm to the reader. Use the name of the algorithm or model as the subsection title. 
Present a detailed description about 
1. Use one paragraph to describe what analysis model or algorithm used in this section, 
2. Generate a enumerate list to describe the step-by-step way of how to use this model or algorithm to perform ananlysis. When generating the step-by-step description, you can revise the text provided to you, and reference online resources, 
3. Generate equations to present the calculation formular used in this algorithm or model using the latex equation package.
\subsection
Roofline Plot
Insert an image for the roofline plot, FP8. the image is saved to the same folder where the report is. the image file name is img/gemmfp8.png, use width=0.8\textwidth to control the width of the fiture.
The roofline plot plots both benchmark and theoretical performance of {target} and {baseline}.
\subsection
Becnhamrk Analysis Result
use expression like "Please refer to Table \ref{tab:xxx} for detailed benchmark results" in this section, replace the xxx with the real label name presented in case field, while you must keep the 'tab' string to avoid build error, you don't need to generate the table here, another process will do it.
generate paragraph with highlighted numbers to explain the findings and speedup from {baseline} to {target}, describe the findings for math-limited kernels and memory-limited kernels separately.


[case]

