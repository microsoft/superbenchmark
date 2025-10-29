[knowledge]

knowledge 1
SuperBench, developed by Microsoft Research, is an open-sourced standard tool for machine learning system performance validation.

knowledge 2
We run SuperBench end-to-end model benchmark on {target}, and {baseline} SKUs.
We cover different models including transformer based models (bert, gpt-2), cnn models (densenet, resnet, vgg), and lstm.
The precision is FP32 and FP16.

The comparison result is presented using the following metrics:
\begin{equation}
    \mathit{Speedup} = \frac{\mathit{TrainingThroughput}_{\mathit{{target}}}}{\mathit{TrainingThroughput}_{\mathit{{baseline}}}}
\end{equation}
A value of greater than 1 indicates {target} performs better.

knowledge 3
Introduction of SuperBench
Features#
SuperBench is a validation and profiling tool for AI infrastructure, which supports:
	• AI infrastructure validation and diagnosis
	• Distributed validation tools to validate hundreds or thousands of servers automatically
	• Consider both raw hardware and E2E model performance with ML workload patterns
	• Build a contract to identify hardware issues
	• Provide infrastructural-oriented criteria as Performance/Quality Gates for hardware and system release
	• Provide detailed performance report and advanced analysis tool
	• AI workload benchmarking and profiling
	• Provide comprehensive performance comparison between different existing hardware
	• Provide insights for hardware and software co-design
It provides micro-benchmark for primitive computation and communication benchmarking, as well as model-benchmark to measure domain-aware end-to-end deep learning workloads.

Run training or inference tasks with single or half precision for deep learning models, including the following categories:
	• GPT: gpt2-small, gpt2-medium, gpt2-large and gpt2-xl
	• BERT: bert-base and bert-large
	• LSTM
	• CNN, listed in torchvision.models, including:
	• resnet: resnet18, resnet34, resnet50, resnet101, resnet152
	• densenet: densenet121, densenet169, densenet201, densenet161
vgg: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19

