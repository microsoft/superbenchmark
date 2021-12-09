---
id: docker-benchmarks
---

# Docker Benchmarks

## ROCm ONNXRuntime Model Benchmarks

### `ort-models`

#### Introduction

Run the rocm onnxruntime model training benchmarks packaged in docker `superbench/benchmark:rocm4.3.1-onnxruntime1.9.0` which includes Bert-large, Distilbert-base, GPT-2, facebook/Bart-large and Roberta-large.

#### Metrics

| Name                                                                   | Unit                   | Description                                               |
|------------------------------------------------------------------------|------------------------|-----------------------------------------------------------|
| onnxruntime-ort-models/bert_large_uncased_ngpu_1_train_throughput      | throughput (samples/s) | The throughput of bert large uncased model on 1 GPU.      |
| onnxruntime-ort-models/bert_large_uncased_ngpu_8_train_throughput      | throughput (samples/s) | The throughput of bert large uncased model on 8 GPU.      |
| onnxruntime-ort-models/distilbert_base_uncased_ngpu_1_train_throughput | throughput (samples/s) | The throughput of distilbert base uncased model on 1 GPU. |
| onnxruntime-ort-models/distilbert_base_uncased_ngpu_8_train_throughput | throughput (samples/s) | The throughput of distilbert base uncased model on 8 GPU. |
| onnxruntime-ort-models/gpt2_ngpu_1_train_throughput                    | throughput (samples/s) | The throughput of gpt2 model on 1 GPU.                    |
| onnxruntime-ort-models/gpt2_ngpu_8_train_throughput                    | throughput (samples/s) | The throughput of gpt2 model on 8 GPU.                    |
| onnxruntime-ort-models/facebook_bart_large_ngpu_1_train_throughput     | throughput (samples/s) | The throughput of facebook bart large model on 1 GPU.     |
| onnxruntime-ort-models/facebook_bart_large_ngpu_8_train_throughput     | throughput (samples/s) | The throughput of facebook bart large model on 8 GPU.     |
| onnxruntime-ort-models/roberta_large_ngpu_1_train_throughput           | throughput (samples/s) | The throughput of roberta large model on 1 GPU.           |
| onnxruntime-ort-models/roberta_large_ngpu_8_train_throughput           | throughput (samples/s) | The throughput of roberta large model on 8 GPU.           |
