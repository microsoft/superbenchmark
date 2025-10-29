Your task is to regenerate a section by reading the contents in the 'draft' part.
You should generate the output using {format} format,

[content of output]: /section performance projection method, you must add a reference lable \label{efficiency}
for each paragraph in 'draft' section, repeat it and improve the expression. Don't change stuff between $$, since they math experession for latex.

[draft]
\section{Performance Projection Methodology}\label{sec:efficiency}
In this study, we extrapolate the performance of various workloads on our target GPUs, the AMD MI375X GPUs, based on the performance data from the latest real AMD GPU, the MI300X GPU.

\subsection{Methodology Description}
Our projection methodology fundamentally relies on the principle of FLOPS scaling. The accuracy of our projection results is contingent upon two key factors: the validity of FLOPS scaling and the precision of the profiling data on the current hardware. We first delve into the rationale behind FLOPS scaling, a technique extensively employed in existing research. Subsequently, we discuss strategies to enhance the fidelity of profiling.

\subsubsection{FLOPS Scaling}
The fundamental assumption of FLOPS scaling is its validity, given that the hardware efficiency remains constant. Our empirical data suggest a consistent level of hardware efficiency. This is further corroborated by information provided by AMD.

Specifically, we employed linear extrapolation where the performance was scaled based on the FLOPS and XGMI performance ratio between the simulated GPU (MI375X) and the real GPU (MI300X). This assumes that the simulated GPU and the real GPU maintain the same hardware efficiency, which is the ratio of the workload's actual performance to the theoretical peak performance from the specification sheet.

In particular, hardware efficiency, denoted as \verb|HE|,  is defined as
\begin{equation}
\label{eq:efficiency-1}
HE = \frac{ExecutionFLOPS}{SpecPeakFLOPS}
\end{equation}

We categorize the kernels of the real GPU's execution into computation kernels and communication kernels.

Assuming that the execution of operators with the same input shape maintains the same HE across GPU platforms from the same vendor, i.e., MI375X and MI300X, we can deduce the overall computation kernel latency on the simulated GPU as follows.

\begin{equation}
\label{eq:efficiency-2}
\begin{aligned}
latency\_compute_{simulated} & = \sum{latency\_kernel_i} = \sum{\frac{kernel_{i}\_total\_flops}{Execution_{i}FLOPS_{simulated}}} \\
& = \sum{\frac{kernel_{i}\_total\_flops}{{HE_{i} \cdot SpecPeakFLOPS_{simulated}}}}\\
& = \sum{\frac{kernel_{i}\_total\_flops}{\frac{Execution_{i}FLOPS_{real}}{SpecPeakFLOPS_{real}} \cdot SpecPeakFLOPS_{simulated}}}\\
& =  \frac{SpecPeakFLOPS_{real}}{SpecPeakFLOPS_{simulated}} \cdot \sum{\frac{kernel_{i}\_total\_flops}{Execution_{i}FLOPS_{real}}}
\end{aligned}
\end{equation}

This equation essentially allows us to deduce the performance of computation kernels on simulated GPUs as follows:

\begin{equation}
\label{eq:efficiency-3}
latency\_compute_{simulated} = \frac{SpecPeakFLOPS_{real}}{SpecPeakFLOPS_{simulated}} \cdot latency\_compute_{real}
\end{equation}

By applying the same methodology, we can project the performance of communication kernels as follows:

\begin{equation}
\label{eq:efficiency-4}
latency\_communication_{simulated} = \frac{SpecPeakNetworkBW_{real}}{SpecPeakNetworkBW_{simulated}} \cdot latency\_communication_{real}
\end{equation}

We then aggregate all kernels based on the trace to project the end-to-end workload performance.

The workload performance projection on MI355X, also adopts the same approach by linear extrapolation from the profiled workload performance number from MI300X.

\subsubsection{Profiling Fidelity}
In terms of profiling accuracy, we consider several strategies to enhance its fidelity.
\begin{itemize}
\item Profiling inherently introduces overhead.
    \begin{itemize}
    \item Profiling is conducted repetitively (i.e., at least 50 steps) and the profiled data we use will exclude any unreasonable outliers.
    \item We employ different profiling tools. Here, we consider both existing profiling tools (e.g., PyTorch Profiler, Nsight system, and ROCProfiler) and in-line timing event injection to measure the durations of the execution kernels. We compare the summation of profiled kernel durations with end-to-end latency (without profiling) to identify the best profiling approach.
    \item For small input data sizes, where profiling overhead can dominate, we limit the number of execution kernels profiled per iteration and increase the total number of iterations to ensure full kernel coverage.
    \end{itemize}
\item Gaps between consecutive execution kernels are depicted. In Symphora, we have identified there are gaps between consecutive execution kernels, which cannot be explicitly captured with any profiling tools or timing utilities. We extract such gaps in the profiled trace and adjust our estimation to account for them on newer hardware.
\end{itemize}


