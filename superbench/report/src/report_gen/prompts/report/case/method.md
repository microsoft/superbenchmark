your task is to generate an appendix section to present some meta information about the benchmark report.

output format:
You should generate the output using {format} format.
This is a section with section title 'Methodology of Auto Performance Evaluation'

[content of output]
/section
Automated Performance Evaluation Pipeline
AIMicius introduced a fully automated performance evaluation pipeline for machine learning infrastructure. This pipeline encompasses three crucial phases: automatic node benchmarking, automatic performance optimization, and automatic report generation. The implementation of this automated pipeline has significantly reduced the time required for Azure VM performance validation iteration from over a month to an optimal of 2 days.
The table below compares the time and effort required for manual and automated pipeline tasks:

\begin{table}
\center
  \caption{Comparison Between Automated Effort and Manual Effort}
  \label{tab:method-1}
  \begin{tabular}{l c c}
\toprule
Task & \makecell[l]{Manual Effort on \\ MI200 EV 9234 IFWI SKU} & \makecell[l]{Automated Effort (Optimal) \\ on MI300x A1 BKC35 SKU} \\
\midrule
Performance Benchmark & 15 business days & 18 hours \\
Performance Optimization & 15 business days & 4 hours \\
Benchmark Reporting & 5 business days & 3 hours \\
\bottomrule
\end{tabular}
\end{table}

For more detailed information, please refer to the accompanying comprehensive report. The following section introduces several technologies utilized in the report generation process.

/subsection
Report Generation Method
The analysis of results and generation of report content are carried out by Project AIMicius, which is powered by LLM. 
A sub-component has been designed to address scenarios where users seek information on how to evaluate the performance of a machine learning data center infrastructure, to obtain a comprehensive overview or to assess how well the infrastructure performs on specific workloads. AIMicius utilizes raw benchmark data as input and generates the entire report using functions provided by the system. Below, we outline some of the key concepts behind this process.

/subsubsection
Prompt Engineering Techniques
We employ a structured prompt template created by human experts. This well-organized template effectively generalizes similar user cases and ensures accuracy and reliability, as the expectations from GPT can be clearly defined, and the knowledge and tools required for GPT to complete the task can be well understood. 
Zero-shot examples of step-by-step instructions are also utilized, which are useful for guiding GPT in performing specific user-defined tasks, such as writing Python code for a particular algorithm or executing a widely adopted report writing process. 
Additionally, we use limited response token count guidance in the prompts, which helps GPT concentrate on delivering essential responses.
We ask GPT to generate code for analyzing the raw data table, as the raw data length does not meet the input token limit requirement.

/subsubsection
Feeding Domain Knowledge
Additional knowledge is supplied to GPT to perform the task. System performance evaluation methods are provided, as the general GPT model lacks this information. For instance, we need to use few-shot prompting so that GPT learns when to use the roofline model, even though it is already familiar with it. Information from recent years is also necessary, as GPT-4 training data does not encompass data from the past two years. This includes specifications of recently released hardware, new workload structures, and more.

/subsubsection
Minimal Human Interference
AIMicius generates the entire content of the report (excluding the Appendix section) in a few steps. It analyzes the data for each benchmark, compares it to the theoretical value or reference SKU value, highlights key performance improvements or drawbacks, and summarizes all benchmarks to produce the executive summary and conclusion. Human intervention is only required to execute benchmarking on models that mimics procudtion scenario workload, evaluate the correctness of all data values, as well as the correctness and clarity of the report.
