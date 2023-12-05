---
id: model-benchmarks
---

# Model Benchmarks

## PyTorch Model Benchmarks

### `model-benchmarks`

#### Introduction

Run training or inference tasks with single or half precision for deep learning models,
including the following categories:
* GPT: gpt2-small, gpt2-medium, gpt2-large and gpt2-xl
* BERT: bert-base and bert-large
* LSTM
* CNN, listed in [`torchvision.models`](https://pytorch.org/vision/0.8/models.html), including:
  * resnet: resnet18, resnet34, resnet50, resnet101, resnet152
  * resnext: resnext50_32x4d, resnext101_32x8d
  * wide_resnet: wide_resnet50_2, wide_resnet101_2
  * densenet: densenet121, densenet169, densenet201, densenet161
  * vgg: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19
  * mnasnet: mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
  * mobilenet: mobilenet_v2
  * shufflenet: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
  * squeezenet: squeezenet1_0, squeezenet1_1
  * others: alexnet, googlenet, inception_v3

For inference, supported percentiles include
50<sup>th</sup>, 90<sup>th</sup>, 95<sup>th</sup>, 99<sup>th</sup>, and 99.9<sup>th</sup>.

**New: Support fp8_hybrid and fp8_e4m3 precision for BERT models.**

#### Metrics

| Name                                                                                    | Unit                   | Description                                                                  |
|-----------------------------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------|
| model-benchmarks/pytorch-${model_name}/${precision}_train_step_time                     | time (ms)              | The average training step time with fp32/fp16 precision.                     |
| model-benchmarks/pytorch-${model_name}/${precision}_train_throughput                    | throughput (samples/s) | The average training throughput with fp32/fp16 precision per GPU.            |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_step_time                 | time (ms)              | The average inference step time with fp32/fp16 precision.                    |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_throughput                | throughput (samples/s) | The average inference throughput with fp32/fp16 precision.                   |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_step_time\_${percentile}  | time (ms)              | The n<sup>th</sup> percentile inference step time with fp32/fp16 precision.  |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_throughput\_${percentile} | throughput (samples/s) | The n<sup>th</sup> percentile inference throughput with fp32/fp16 precision. |


## Megatron Model benchmarks

### `megatron-gpt`

#### Introduction

Run GPT pretrain tasks with float32, float16, bfloat16 precisions with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed).

`tips: batch_size in this benchmark represents global batch size, the batch size on each GPU instance is micro_batch_size.`

#### Metrics
| Name                                              | Unit                   | Description                                             |
|---------------------------------------------------|------------------------|---------------------------------------------------------|
| megatron-gpt/${precision}_train_step_time         | time (ms)              | The average training step time per iteration.           |
| megatron-gpt/${precision}_train_throughput        | throughput (samples/s) | The average training throughput per iteration.          |
| megatron-gpt/${precision}_train_tflops            | tflops/s               | The average training tflops per second per iteration.   |
| megatron-gpt/${precision}_train_mem_allocated     | GB                     | The average GPU memory allocated per iteration.         |
| megatron-gpt/${precision}_train_max_mem_allocated | GB                     | The average maximum GPU memory allocated per iteration. |

