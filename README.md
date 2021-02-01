# SuperBenchmark

[![Build Status](https://dev.azure.com/msrasrg/SuperBenchmark/_apis/build/status/microsoft.superbenchmark?branchName=dev)](https://dev.azure.com/msrasrg/SuperBenchmark/_build?definitionId=77)
[![Lint](https://github.com/microsoft/superbenchmark/workflows/Lint/badge.svg)](https://github.com/microsoft/superbenchmark/actions?query=workflow%3ALint)


SuperBench is a benchmarking and diagnosis tool for AI infrastructure,
which supports:
* Comprehensive AI infrastructure validation
    * Distribute validation tools to validate hundreds or thousands of severs automatically
    * Consider both raw hardware and E2E model performance with ML workload patterns
    * Provide a fast and accurate way to detect and locate hardware problems
    * Performance/Quality Gates for hardware and system release
* Benchmarking with typical AI workload patterns
    * Provide comprehensive performance comparison between different existing hardware
    * Give a better understanding for new DL software & hardware
* Detailed performance analysis and diagnosis
    * Provide detailed performance report and advanced analysis tool   

It includes micro-benchmark for primitive computation and communication benchmarking,
and model-benchmark to measure domain-aware end-to-end deep learning workloads.


## Installation

### Docker Image

TODO

### From Source

If you are installing from source, you will need Python 3.6 or later.

```sh
# get source code
git clone https://github.com/microsoft/superbenchmark
cd superbenchmark

# install superbench
python3 setup.py install
```


## Usage

TODO


## Developer Guide

### Set Up

```sh
# get dev branch code
git clone -b dev https://github.com/microsoft/superbenchmark
cd superbenchmark

# install superbench
python3 -m pip install -e .[dev,test]
```

### Lint and Test

```sh
# format code using yapf
python3 setup.py format

# check code style with mypy and flake8
python3 setup.py lint

# run all unit tests
python3 setup.py test
```

### Submit a Pull Request

Please install `pre-commit` before `git commit` to run all pre-checks.

```sh
pre-commit install
```

Pull requests should be submitted to `dev` branch.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
