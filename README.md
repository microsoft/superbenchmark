# SuperBench

[![Build Image](https://github.com/microsoft/superbenchmark/workflows/Build%20Image/badge.svg)](https://github.com/microsoft/superbenchmark/actions/workflows/build-image.yml)
[![Codecov](https://codecov.io/gh/microsoft/superbenchmark/branch/main/graph/badge.svg?token=DDiDLW7pSd)](https://codecov.io/gh/microsoft/superbenchmark)
[![Website](https://img.shields.io/website?down_color=lightgrey&url=https%3A%2F%2Faka.ms%2Fsuperbench)](https://aka.ms/superbench)
[![Latest Release](https://img.shields.io/github/release/microsoft/superbenchmark.svg)](https://github.com/microsoft/superbenchmark/releases/latest)
[![Docker Pulls](https://img.shields.io/docker/pulls/superbench/superbench.svg)](https://hub.docker.com/r/superbench/superbench/tags)
[![License](https://img.shields.io/github/license/microsoft/superbenchmark.svg)](LICENSE)

| Azure Pipelines          | Build Status                                                                                                                                                                                                            |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| cpu-unit-test            | [![Build Status](https://dev.azure.com/msrasrg/SuperBenchmark/_apis/build/status/cpu-unit-test?branchName=main)](https://dev.azure.com/msrasrg/SuperBenchmark/_build/latest?definitionId=77&branchName=main)            |
| cuda-unit-test           | [![Build Status](https://dev.azure.com/msrasrg/SuperBenchmark/_apis/build/status/cuda-unit-test?branchName=main)](https://dev.azure.com/msrasrg/SuperBenchmark/_build/latest?definitionId=80&branchName=main)           |
| ansible-integration-test | [![Build Status](https://dev.azure.com/msrasrg/SuperBenchmark/_apis/build/status/ansible-integration-test?branchName=main)](https://dev.azure.com/msrasrg/SuperBenchmark/_build/latest?definitionId=82&branchName=main) |

__SuperBench__ is a validation and profiling tool for AI infrastructure.

📢 [v0.11.0](https://github.com/microsoft/superbenchmark/releases/tag/v0.11.0) has been released!

## _Check [aka.ms/superbench](https://aka.ms/superbench) for more details._

## Citations

To cite SuperBench in your publications:

```bib
@inproceedings {superbench,
	author = {Yifan Xiong and Yuting Jiang and Ziyue Yang and Lei Qu and Guoshuai Zhao and Shuguang Liu and Dong Zhong and Boris Pinzur and Jie Zhang and Yang Wang and Jithin Jose and Hossein Pourreza and Jeff Baxter and Kushal Datta and Prabhat Ram and Luke Melton and Joe Chau and Peng Cheng and Yongqiang Xiong and Lidong Zhou},
	title = {{SuperBench}: Improving Cloud {AI} Infrastructure Reliability with Proactive Validation},
	booktitle = {2024 USENIX Annual Technical Conference (USENIX ATC 24)},
	year = {2024},
	isbn = {978-1-939133-41-0},
	address = {Santa Clara, CA},
	pages = {835--850},
	url = {https://www.usenix.org/conference/atc24/presentation/xiong},
	publisher = {USENIX Association},
	month = jul
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
