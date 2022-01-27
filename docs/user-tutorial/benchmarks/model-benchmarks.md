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

#### Metrics

| Name                                                                            | Unit                   | Description                                                               |
|---------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------|
| model-benchmarks/pytorch-${model_name}/fp32_train_step_time                     | time (ms)              | The average training step time with single precision.                     |
| model-benchmarks/pytorch-${model_name}/fp32_train_throughput                    | throughput (samples/s) | The average training throughput with single precision.                    |
| model-benchmarks/pytorch-${model_name}/fp32_inference_step_time                 | time (ms)              | The average inference step time with single precision.                    |
| model-benchmarks/pytorch-${model_name}/fp32_inference_throughput                | throughput (samples/s) | The average inference throughput with single precision.                   |
| model-benchmarks/pytorch-${model_name}/fp32_inference_step_time\_${percentile}  | time (ms)              | The n<sup>th</sup> percentile inference step time with single precision.  |
| model-benchmarks/pytorch-${model_name}/fp32_inference_throughput\_${percentile} | throughput (samples/s) | The n<sup>th</sup> percentile inference throughput with single precision. |
| model-benchmarks/pytorch-${model_name}/fp16_train_step_time                     | time (ms)              | The average training step time with half precision.                       |
| model-benchmarks/pytorch-${model_name}/fp16_train_throughput                    | throughput (samples/s) | The average training throughput with half precision.                      |
| model-benchmarks/pytorch-${model_name}/fp16_inference_step_time                 | time (ms)              | The average inference step time with half precision.                      |
| model-benchmarks/pytorch-${model_name}/fp16_inference_throughput                | throughput (samples/s) | The average inference throughput with half precision.                     |
| model-benchmarks/pytorch-${model_name}/fp16_inference_step_time\_${percentile}  | time (ms)              | The n<sup>th</sup> percentile inference step time with half precision.    |
| model-benchmarks/pytorch-${model_name}/fp16_inference_throughput\_${percentile} | throughput (samples/s) | The n<sup>th</sup> percentile inference throughput with half precision.   |
