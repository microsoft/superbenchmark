---
id: contributing
---

# Contributing

## Contributor License Agreement

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## How to Contribute

### Contribute New Feature

SuperBenchmark is an open-source project. Your participation and contribution are highly appreciated. There are several important things you need know before contributing new feature to this project:

#### What content can be added to SuperBenchmark

1. Bug fixes for existing features.
2. New features for benchmark module (micro-benchmark, model-benchmark, etc.)

   If you would like to contribute a new feature on SuperBenchmark, please submit your proposal first. In [GitHub Issues](https://github.com/microsoft/superbenchmark/issues) module, choose `Enhancement Request` to finish the submission. If the proposal is accepted, you can submit pull requests to origin `main` branch.

#### Contribution steps

If you would like to contribute to the project, please follow below steps of joint development on GitHub.

1. `Fork` the repo first to your personal GitHub account.
2. Checkout from main branch for feature development.
3. When you finish the feature, please fetch the latest code from origin repo, merge to your branch and resolve conflict.
4. Submit pull requests to origin main branch.
5. Please note that there might be comments or questions from reviewers. It will need your help to update the pull request.


### Contribute Benchmark Results

If you want to contribute benchmark results run by specified SuperBench version, please follow below guidelines.

#### Where to submit

All the results are stored under [superbench-results](https://github.com/microsoft/superbench-results) repository. The directory structure is as follows. Please create `<your-benchmark-folder>` to submit results.

```
superbench-results
  ├── v0.2
  │   └── your-benchmark-foldername
  │       ├── LICENSE.md
  │       ├── README.md
  │       ├── configs
  │       │   ├── config1.yaml
  │       │   └── config2.yaml
  │       ├── results
  │       │   ├── result1.json
  │       │   └── result2.json
  │       └── systems
  │           ├── system1.json
  │           └── system2.json
  └── v0.3
      └── your-benchmark-foldername
          ├── LICENSE.md
          ├── README.md
          ├── configs
          │   ├── config1.yaml
          │   └── config2.yaml
          ├── results
          │   ├── result1.json
          │   └── result2.json
          └── systems
              ├── system1.json
              └── system2.json
```

#### Files to provide

Besides `README` and `LICENSE` file, you should provide at least three benchmarking related files.
* `system.json`: This file lists all the system configurations in json format.

  You can get the system info automatically by executing `system_info.py` using below command. The file is under `superbench/tools` folder.
  ```
  python system_info.py
  ```
* `config.yaml`: This file is the config file to run benchmarks. Click [here](../getting-started/configuration.md) to learn the details.
* `result.json`: This file contains the results run by SuperBench with system configuations listed in `system.json` file.
