# SuperBenchmark

[![Build Status](https://dev.azure.com/msrasrg/SuperBenchmark/_apis/build/status/microsoft.superbenchmark?branchName=dev)](https://dev.azure.com/msrasrg/SuperBenchmark/_build?definitionId=77)
[![Lint](https://github.com/microsoft/superbenchmark/workflows/Lint/badge.svg)](https://github.com/microsoft/superbenchmark/actions?query=workflow%3ALint)


SuperBench is a benchmarking and diagnosis tool for AI infrastructure,
which supports:
* Comprehensive AI infrastructure validation
    * Distributed validation tools to validate hundreds or thousands of servers automatically
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

### Using Python

System requirements:
* Python: Python 3.6 or later, pip 18.0 or later
* Platform: Ubuntu 16.04 or later (64-bit), Windows 10 (64-bit) with WSL2

Check whether Python environment is already configured:
```sh
# check Python version
python3 --version
# check pip version
python3 -m pip --version
```
If not, install the followings:
* [Python](https://www.python.org/)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [venv](https://docs.python.org/3/library/venv.html)

It's recommended to use a virtual environment (optional):
```sh
# create a new virtual environment
python3 -m venv --system-site-packages ./venv
# activate the virtual environment
source ./venv/bin/activate

# exit the virtual environment later
# after you finish running superbench
deactivate
```

Then install superbench through either PyPI binary or from source:

1. PyPI Binary

    TODO

2. From Source

    ```sh
    # get source code
    git clone https://github.com/microsoft/superbenchmark
    cd superbenchmark

    # install superbench
    python3 -m pip install .
    ```

### Using Docker

TODO


## Usage

TODO


## Developer Guide

Follow [Installation using Python](#using-python) and
use [`dev` branch](https://github.com/microsoft/superbenchmark/tree/dev).

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

Pull requests should be submitted to [`dev` branch](https://github.com/microsoft/superbenchmark/tree/dev).


## Contributing

### Contributor License Agreement

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Contributing principles

SuperBenchmark is an open benchmarking project. Contributions are highly welcome. If you want to contribute to the project, follow things should be known.

#### What's content can be added to SuperBenchmark

1. Bug fixes of the current features.
2. New benchmark (micro-benchmark / model-benchmark) module impleamentation.
   
   If you want to contribute new benchmark, please submit your proposal first. In [GitHub Issues](https://github.com/microsoft/superbenchmark/issues) mudule, choose `Enhancement Request` to finish the submission. If the proposal is accepted, you can submit pull request of the implemenatation.

#### Contribution rules

If you want to contribut to the project, please follow the basic rules of joint development on GitHub.

1. `Fork` the repo first to your personal GitHub account.
2. Check out to `dev` branch for feature development.
3. When you finished the feature, fetch the latest code from origin repo, merge to your branch and resolve conflict.
4. Submit pull request to origin `dev` branch.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
