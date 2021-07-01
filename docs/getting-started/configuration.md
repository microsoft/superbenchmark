---
id: configuration
---

# Configuration

## SuperBench config

SuperBench uses a [YAML](https://yaml.org/spec/1.2/spec.html) config file to configure the details of benchmarkings,
including which benchmark to run, which distributing mode to choose, which parameter to use, etc.

Here's what default config file looks like.

```yaml title="superbench/config/default.yaml"
# SuperBench Config
superbench:
  enable: null
  var:
    default_local_mode: &default_local_mode
      enable: true
      modes:
        - name: local
          proc_num: 8
          prefix: CUDA_VISIBLE_DEVICES={proc_rank}
          parallel: yes
    default_pytorch_mode: &default_pytorch_mode
      enable: true
      modes:
        - name: torch.distributed
          proc_num: 8
          node_num: 1
      frameworks:
        - pytorch
    common_model_config: &common_model_config
      duration: 0
      num_warmup: 16
      num_steps: 128
      precision:
        - float32
        - float16
      model_action:
        - train
  benchmarks:
    kernel-launch:
      <<: *default_local_mode
    gemm-flops:
      <<: *default_local_mode
    cudnn-function:
      <<: *default_local_mode
    cublas-function:
      <<: *default_local_mode
    matmul:
      <<: *default_local_mode
      frameworks:
        - pytorch
    sharding-matmul:
      <<: *default_pytorch_mode
    computation-communication-overlap:
      <<: *default_pytorch_mode
    gpt_models:
      <<: *default_pytorch_mode
      models:
        - gpt2-small
        - gpt2-large
      parameters:
        <<: *common_model_config
        batch_size: 4
    bert_models:
      <<: *default_pytorch_mode
      models:
        - bert-base
        - bert-large
      parameters:
        <<: *common_model_config
        batch_size: 8
    lstm_models:
      <<: *default_pytorch_mode
      models:
        - lstm
      parameters:
        <<: *common_model_config
        batch_size: 128
    resnet_models:
      <<: *default_pytorch_mode
      models:
        - resnet50
        - resnet101
        - resnet152
      parameters:
        <<: *common_model_config
        batch_size: 128
    densenet_models:
      <<: *default_pytorch_mode
      models:
        - densenet169
        - densenet201
      parameters:
        <<: *common_model_config
        batch_size: 128
    vgg_models:
      <<: *default_pytorch_mode
      models:
        - vgg11
        - vgg13
        - vgg16
        - vgg19
      parameters:
        <<: *common_model_config
        batch_size: 128
```

By default, all benchmarks in default configuration will be run if you don't specify customized configuration.

If you want to have a quick try, you can modify this config a little bit. For example, only run resnet101 model.
1. copy the default config to a file named `resnet.yaml` in current path.
  ```bash
  cp superbench/config/default.yaml resnet.yaml
  ```
2. enable only `resnet_models` in the config and remove other models except resnet101 under `benchmarks.resnet_models.models`.
  ```yaml {3,11} title="resnet.yaml"
  # SuperBench Config
  superbench:
    enable: ['resnet_models']
    var:
  # ...
  # omit the middle part
  # ...
      resnet_models:
        <<: *default_pytorch_mode
        models:
          - resnet101
        parameters:
          <<: *common_model_config
          batch_size: 128
  ```

## Ansible Inventory

SuperBench leverages [Ansible](https://docs.ansible.com/ansible/latest/) to run benchmarking workloads on managed nodes,
you need to provide an [inventory](https://docs.ansible.com/ansible/latest/user_guide/intro_inventory.html) file
to configure host list for managed nodes.

Here're some basic examples as your starting point.
* One managed node, same node as control node.
  ```ini title="local.ini"
  [all]
  localhost ansible_connection=local
  ```
* Two managed nodes, one is control node and the other can be remote accessed.
  ```ini title="mix.ini"
  [all]
  localhost ansible_connection=local
  10.0.0.100 ansible_user=username ansible_ssh_private_key_file=id_rsa
  ```
* Eight managed nodes, all can be accessed remotely.
  ```ini title="remote.ini"
  [all]
  10.0.0.[100:103]
  10.0.0.[200:203]

  [all:vars]
  ansible_user=username
  ansible_ssh_private_key_file=id_rsa
  ```
