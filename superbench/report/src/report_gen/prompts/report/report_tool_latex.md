[tool]

tool 1
When you generate latex format report title and headers, use below rule: in titles and headers of a report, it's common to capitalize all major words, including verbs, nouns, adjectives, adverbs, and pronouns. However, you're correct that smaller words like 'a', 'an', 'and', 'the', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by' are often left in lowercase. 

tool 2
When you generate itemize list to highlight key findings and suggestions, please always include the quantitive numbers from a comparison result. Convert the numbers into a times value. e.g., use 0.5X to represent if testbed A performance is 0.5 or 50% of testbed B, use 2.4X to present if testbed A performance is 2.4 times or 240% of testbed B. When you generate tables, you don't need to add the X mark.

tool 3
When you compare values of metrics, use these rules:
rule (1), if a metric meaning is throughput, bandwidth or FLOPS, a larger value means better perofrmance, if a metric meaning is time or latency, a smaller value means better performance
rule (2), a difference smaller than 0.05X or smaller than 5% means their performance is almost the same.

tool 4
When you need to generate latex tables, you must use this table format:
place caption above table
use left alighment for all columns
use \toprule before header
use \midrule after header
use \bottomrule after last row

example:
\begin{table}[h]
\centering
\caption{value of title}
\label{value of lable}
\begin{tabular}{l l}
\toprule
header 1 & \makecell[l]{string 1 of header 2 \\ string 2 of header 2} \\
\midrule
row 1 column 1 & row 1 column 2 \\
row 2 column 1 & row 2 column 2 \\
\bottomrule
\end{tabular}
\end{table}

tool 5
When you need to present a percentage value, you must use '\%' instead of '%', otherwise it will cause latex compile error.

tool 6
footnotes in draft or case part must be preserved.

tool 7
avoid using & to express the meaning of and, use the word and instead to avoid latex syntax error during compilation