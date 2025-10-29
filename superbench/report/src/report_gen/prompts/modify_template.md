[task]
Your task is to userstand the template modification requirement from the user message, then modify the prompt template and output the whole modified template using below format:
```text

```
Example 1:
user input = add one more subsection before the subsection evaluation to describe to test tool mlc
prompt template = 
```
[specification]

[content of the report]

The content of this section of report should include:
\section
use the section title of 'Typical Small Model Training Performance'

\subsection
Evaluate the results
```
output should be
```text
[specification]

[content of the report]

The content of this section of report should include:
\section
use the section title of 'Typical Small Model Training Performance'

\subsection
Introduce mlc's function and capability.

\subsection
Evaluate the results
```

