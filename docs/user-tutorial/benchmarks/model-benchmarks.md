---
id: model-benchmarks
---

# Model Benchmarks

## PyTorch Model Benchmarks

### `gpt_models`

#### Introduction

Run training or inference tasks with single or half precision for GPT models,
including gpt2-small, gpt2-medium, gpt2-large and gpt2-xl.

#### Metrics

| Name                                                          | Unit                   | Description                                 |
|---------------------------------------------------------------|------------------------|---------------------------------------------|
| gpt_models/pytorch-${model_name}/steptime_train_float32       | time (ms)              | Train with single precision step time.      |
| gpt_models/pytorch-${model_name}/throughput_train_float32     | throughput (samples/s) | Train with single precision throughput.     |
| gpt_models/pytorch-${model_name}/steptime_inference_float32   | time (ms)              | Inference with single precision step time.  |
| gpt_models/pytorch-${model_name}/throughput_inference_float32 | throughput (samples/s) | Inference with single precision throughput. |
| gpt_models/pytorch-${model_name}/steptime_train_float16       | time (ms)              | Train with half precision step time.        |
| gpt_models/pytorch-${model_name}/throughput_train_float16     | throughput (samples/s) | Train with half precision throughput.       |
| gpt_models/pytorch-${model_name}/steptime_inference_float16   | time (ms)              | Inference with half precision step time.    |
| gpt_models/pytorch-${model_name}/throughput_inference_float16 | throughput (samples/s) | Inference with half precision throughput.   |

### `bert_models`

#### Introduction

Run training or inference tasks with single or half precision for BERT models, including bert-base and bert-large.

#### Metrics

| Name                                                           | Unit                   | Description                                 |
|----------------------------------------------------------------|------------------------|---------------------------------------------|
| bert_models/pytorch-${model_name}/steptime_train_float32       | time (ms)              | Train with single precision step time.      |
| bert_models/pytorch-${model_name}/throughput_train_float32     | throughput (samples/s) | Train with single precision throughput.     |
| bert_models/pytorch-${model_name}/steptime_inference_float32   | time (ms)              | Inference with single precision step time.  |
| bert_models/pytorch-${model_name}/throughput_inference_float32 | throughput (samples/s) | Inference with single precision throughput. |
| bert_models/pytorch-${model_name}/steptime_train_float16       | time (ms)              | Train with half precision step time.        |
| bert_models/pytorch-${model_name}/throughput_train_float16     | throughput (samples/s) | Train with half precision throughput.       |
| bert_models/pytorch-${model_name}/steptime_inference_float16   | time (ms)              | Inference with half precision step time.    |
| bert_models/pytorch-${model_name}/throughput_inference_float16 | throughput (samples/s) | Inference with half precision throughput.   |

### `lstm_models`

#### Introduction

Run training or inference tasks with single or half precision for one bidirectional LSTM model.

#### Metrics

| Name                                                  | Unit                   | Description                                 |
|-------------------------------------------------------|------------------------|---------------------------------------------|
| lstm_models/pytorch-lstm/steptime_train_float32       | time (ms)              | Train with single precision step time.      |
| lstm_models/pytorch-lstm/throughput_train_float32     | throughput (samples/s) | Train with single precision throughput.     |
| lstm_models/pytorch-lstm/steptime_inference_float32   | time (ms)              | Inference with single precision step time.  |
| lstm_models/pytorch-lstm/throughput_inference_float32 | throughput (samples/s) | Inference with single precision throughput. |
| lstm_models/pytorch-lstm/steptime_train_float16       | time (ms)              | Train with half precision step time.        |
| lstm_models/pytorch-lstm/throughput_train_float16     | throughput (samples/s) | Train with half precision throughput.       |
| lstm_models/pytorch-lstm/steptime_inference_float16   | time (ms)              | Inference with half precision step time.    |
| lstm_models/pytorch-lstm/throughput_inference_float16 | throughput (samples/s) | Inference with half precision throughput.   |

### `cnn_models`

#### Introduction

Run training or inference tasks with single or half precision for CNN models listed in
[`torchvision.models`](https://pytorch.org/vision/0.8/models.html), including:
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

#### Metrics

| Name                                                          | Unit                   | Description                                 |
|---------------------------------------------------------------|------------------------|---------------------------------------------|
| cnn_models/pytorch-${model_name}/steptime_train_float32       | time (ms)              | Train with single precision step time.      |
| cnn_models/pytorch-${model_name}/throughput_train_float32     | throughput (samples/s) | Train with single precision throughput.     |
| cnn_models/pytorch-${model_name}/steptime_inference_float32   | time (ms)              | Inference with single precision step time.  |
| cnn_models/pytorch-${model_name}/throughput_inference_float32 | throughput (samples/s) | Inference with single precision throughput. |
| cnn_models/pytorch-${model_name}/steptime_train_float16       | time (ms)              | Train with half precision step time.        |
| cnn_models/pytorch-${model_name}/throughput_train_float16     | throughput (samples/s) | Train with half precision throughput.       |
| cnn_models/pytorch-${model_name}/steptime_inference_float16   | time (ms)              | Inference with half precision step time.    |
| cnn_models/pytorch-${model_name}/throughput_inference_float16 | throughput (samples/s) | Inference with half precision throughput.   |
