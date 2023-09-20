#!/bin/bash

source ~/workspace/sb/bin/activate

sb run -f local.ini -c gemm.yaml --output-dir outputs/gemm

sb result summary --data-file outputs/gemm/results-summary.jsonl --rule-file gemm-rule.yaml --output-dir outputs/gemm --output-file-format md
