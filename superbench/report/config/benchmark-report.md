# Report-Gen Usage Notes

## Required System Packages

Before using Report-Gen, make sure to install the following system packages for LaTeX and PDF generation:

```bash
sudo apt-get install texlive texlive-latex-extra texlive-science texlive-xetex texlive-lang-chinese build-essential fonts-wqy-zenhei biber
```

## Special Pre-processing for GB200

When working with GB200 data, please perform the following pre-processing steps:

1. **Remove `cpu-memory-bw-latency`** from your dataset or input files.
2. **Remove `mem-bw`** from your dataset or input files.

These steps are necessary to ensure correct report generation and avoid errors specific to GB200 cases.
