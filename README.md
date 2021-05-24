# SuperBenchmark

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)
[![Lint](https://github.com/microsoft/superbenchmark/workflows/Lint/badge.svg)](https://github.com/microsoft/superbenchmark/actions?query=workflow%3ALint)
[![Codecov](https://codecov.io/gh/microsoft/superbenchmark/branch/main/graph/badge.svg?token=DDiDLW7pSd)](https://codecov.io/gh/microsoft/superbenchmark)

| Azure Pipelines | Build Status |
| :---: | :---: |
| cpu-unit-test | [![Build Status](https://dev.azure.com/msrasrg/SuperBenchmark/_apis/build/status/microsoft.superbenchmark?branchName=main)](https://dev.azure.com/msrasrg/SuperBenchmark/_build/latest?definitionId=77&branchName=main) |
| gpu-unit-test | [![Build Status](https://dev.azure.com/msrasrg/SuperBenchmark/_apis/build/status/cuda-unit-test?branchName=main)](https://dev.azure.com/msrasrg/SuperBenchmark/_build/latest?definitionId=80&branchName=main) |


**SuperBench** is a validation and profiling tool for AI infrastructure, which supports:

* AI infrastructure validation and diagnosis
    * Distributed validation tools to validate hundreds or thousands of servers automatically
    * Consider both raw hardware and E2E model performance with ML workload patterns
    * Build a contract to identify hardware issues
    * Provide infrastructural-oriented criteria as Performance/Quality Gates for hardware and system release
    * Provide detailed performance report and advanced analysis tool  
  
* AI workload benchmarking and profiling
    * Provide comprehensive performance comparison between different existing hardware
    * Provide insights for hardware and software co-design

It includes micro-benchmark for primitive computation and communication benchmarking,
and model-benchmark to measure domain-aware end-to-end deep learning workloads.

> 🔴 __Note__:
SuperBench is in the early pre-alpha stage for open source, and not ready for general public yet.
If you want to jump in early, you can try building latest code yourself.

## SuperBench capabilities, workflow and benchmarking metrics

The following graphic shows the capabilities provide by SuperBench core framework and its extension.

<img src="imgs/superbench_structure.png">

Benchmarking metrics provided by SuperBench are listed as below.

<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
      </td>
      <td>
        <b>Micro Benchmark</b>
        <img src="imgs/bar.png"/>
      </td>
      <td>
        <b>Model Benchmark</b>
        <img src="imgs/bar.png"/>
      </td>
    </tr>
    <tr valign="top">
      <td align="center" valign="middle">
        <b>Metrics</b>
      </td>
      <td>
        <ul><li><b>Computation Benchmark</b></li>
          <ul><li><b>Kernel Performance</b></li>
            <ul>
              <li>GFLOPS</li>
              <li>TensorCore</li>
              <li>cuBLAS</li>
              <li>cuDNN</li>
            </ul>
          </ul>
          <ul><li><b>Kernel Launch Time</b></li>
            <ul>
              <li>Kernel_Launch_Event_Time</li>
              <li>Kernel_Launch_Wall_Time</li>
            </ul>
          </ul>
          <ul><li><b>Operator Performance</b></li>
            <ul><li>MatMul</li><li>Sharding_MatMul</li></ul>
          </ul>
          <ul><li><b>Memory</b></li>
            <ul><li>H2D_Mem_BW_&lt;GPU ID&gt;</li>
              <li>H2D_Mem_BW_&lt;GPU ID&gt;</li></ul>
          </ul>
        </ul>
        <ul><li><b>Communication Benchmark</b></li>
          <ul><li><b>Device P2P Bandwidth</b></li>
            <ul><li>P2P_BW_Max</li><li>P2P_BW_Min</li><li>P2P_BW_Avg</li></ul>
          </ul>
          <ul><li><b>RDMA</b></li>
            <ul><li>RDMA_Peak</li><li>RDMA_Avg</li></ul>
          </ul>
          <ul><li><b>NCCL</b></li>
            <ul><li>NCCL_AllReduce</li></ul>
            <ul><li>NCCL_AllGather</li></ul>
            <ul><li>NCCL_broadcast</li></ul>
            <ul><li>NCCL_reduce</li></ul>
            <ul><li>NCCL_reduce_scatter</li></ul>
          </ul>
        </ul>
        <ul><li><b>Computation-Communication Benchmark</b></li>
          <ul><li><b>Mul_During_NCCL</b></li><li><b>MatMul_During_NCCL</b></li></ul>
        </ul>
        <ul><li><b>Storage Benchmark</b></li>
          <ul><li><b>Disk</b></li>
            <ul>
              <li>Read/Write</li><li>Rand_Read/Rand_Write</li>
              <li>R/W_Read</li><li>R/W_Write</li><li>Rand_R/W_Read</li><li>Rand_R/W_Write</li>
            </ul>
          </ul>
        </ul>   
      </td>
      <td>
        <ul><li><b>CNN models</b></li>
          <ul>
            <li><b>ResNet</b></li>
              <ul><li>ResNet-50</li><li>ResNet-101</li><li>ResNet-152</li></ul>
          </ul>
          <ul>
            <li><b>DenseNet</b></li>
              <ul><li>DenseNet-169</li><li>DenseNet-201</li></ul>
          </ul>
          <ul>
            <li><b>VGG</b></li>
              <ul><li>VGG-11</li><li>VGG-13</li><li>VGG-16</li><li>VGG-19</li></ul>
          </ul>
          <ul><li><b>Other CNN models</b></li><ul><li>...</li></ul></ul>
        </ul>  
        <ul><li><b>BERT models</b></li>
          <ul><li><b>BERT</b></li><li><b>BERT_LARGE</b></li></ul>
        </ul>
        <ul><li><b>LSTM</b></li></ul>
        <ul><li><b>GPT-2</b></li></ul>
      </td>
    </tr>
  </tbody>
</table>


## Installation

### Using Docker (_Preferred_)

__System Requirements__

* Platform: Ubuntu 18.04 or later (64-bit)
* Docker: Docker CE 19.03 or later

__Install SuperBench__

* Using Pre-Build Images

    ```sh
    docker pull superbench/superbench:dev-cuda11.1.1
    docker run -it --rm \
        --privileged --net=host --ipc=host --gpus=all \
        superbench/superbench:dev-cuda11.1.1 bash
    ```

* Building the Image

    ```sh
    docker build -f dockerfile/cuda11.1.1.dockerfile -t superbench/superbench:dev .
    ```

### Using Python

__System Requirements__

* Platform: Ubuntu 18.04 or later (64-bit); Windows 10 (64-bit) with WSL2
* Python: Python 3.6 or later, pip 18.0 or later

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

__Install SuperBench__

* PyPI Binary

    ```sh
    # not available yet
    ```

* From Source

    ```sh
    # get source code
    git clone https://github.com/microsoft/superbenchmark
    cd superbenchmark

    # install superbench
    python3 -m pip install .
    make postinstall
    ```


## Usage

### Run SuperBench

```sh
# run benchmarks in default settings
sb exec

# use a custom config
sb exec --config-file ./superbench/config/default.yaml
```

### Benchmark Gallary

Please find more benchmark examples [here](examples/benchmarks/).


## Developer Guide

If you want to develop new feature, please follow below steps to set up development environment.

### Check Environment

Follow __[System Requirements](#using-python)__.

### Set Up

```sh
# get latest code
git clone https://github.com/microsoft/superbenchmark
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

Open a pull request to main branch on GitHub.


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

SuperBenchmark is an open-source project. Your participation and contribution are highly appreciated. There are several important things you need know before contributing to this project:

#### What content can be added to SuperBenchmark

1. Bug fixes for existing features.
2. New features for benchmark module (micro-benchmark, model-benchmark, etc.)

   If you would like to contribute a new feature on SuperBenchmark, please submit your proposal first. In [GitHub Issues](https://github.com/microsoft/superbenchmark/issues) module, choose `Enhancement Request` to finish the submission. If the proposal is accepted, you can submit pull requests to origin main branch.

#### Contribution steps

If you would like to contribute to the project, please follow below steps of joint development on GitHub.

1. `Fork` the repo first to your personal GitHub account.
2. Checkout from main branch for feature development.
3. When you finish the feature, please fetch the latest code from origin repo, merge to your branch and resolve conflict.
4. Submit pull requests to origin main branch.
5. Please note that there might be comments or questions from reviewers. It will need your help to update the pull request.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
