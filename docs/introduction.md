---
id: introduction
---

# Introduction

## Features

__SuperBench__ is a validation and profiling tool for AI infrastructure, which supports:

* AI infrastructure validation and diagnosis
  * Distributed validation tools to validate hundreds or thousands of servers automatically
  * Consider both raw hardware and E2E model performance with ML workload patterns
  * Build a contract to identify hardware issues
  * Provide infrastructural-oriented criteria as Performance/Quality Gates for hardware and system release
  * Provide detailed performance report and advanced analysis tool
* AI workload benchmarking and profiling
  * Provide comprehensive performance comparison between different existing hardware
  * Provide insights for hardware and software co-design

It provides micro-benchmark for primitive computation and communication benchmarking,
as well as model-benchmark to measure domain-aware end-to-end deep learning workloads.


## Overview

The following figure shows the capabilities provided by SuperBench core framework and its extension.

![SuperBench Structure](./assets/architecture.svg)
