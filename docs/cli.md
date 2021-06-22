---
id: cli
---

# CLI

SuperBench provides a command line interface to help you use, deploy and run benchmarks.
```
$ sb

   _____                       ____                  _
  / ____|                     |  _ \                | |
 | (___  _   _ _ __   ___ _ __| |_) | ___ _ __   ___| |__
  \___ \| | | | '_ \ / _ \ '__|  _ < / _ \ '_ \ / __| '_ \
  ____) | |_| | |_) |  __/ |  | |_) |  __/ | | | (__| | | |
 |_____/ \__,_| .__/ \___|_|  |____/ \___|_| |_|\___|_| |_|
              | |
              |_|

Welcome to the SB CLI!
```

## SuperBench CLI commands

Below is the full list of SuperBench CLI commands and their usages:

### `sb deploy`

Deploy the SuperBench environments to all managed nodes.
```bash title="SB CLI"
sb deploy [--docker-image]
          [--docker-password]
          [--docker-username]
          [--host-file]
          [--host-list]
          [--host-password]
          [--host-username]
          [--private-key]
```

#### Optional arguments

| Name | Default | Description |
| --- | --- | --- |
| `--docker-image` `-i` | `superbench/superbench` | Docker image URI. |
| `--docker-password` | `None` | Docker registry password if authentication is needed. |
| `--docker-username` | `None` | Docker registry username if authentication is needed. |
| `--host-file` `-f` | `None` | Path to Ansible inventory host file. |
| `--host-list` `-l` | `None` | Comma separated host list. |
| `--host-password` | `None` | Host password or key passphase if needed. |
| `--host-username` | `None` | Host username if needed. |
| `--private-key` | `None` | Path to private key if needed. |

#### Examples

Deploy image `superbench/cuda:11.1` to all nodes in `./host.yaml`:
```bash title="SB CLI"
sb deploy --docker-image superbench/cuda:11.1 --host-file ./host.yaml
```

### `sb exec`

Execute the SuperBench benchmarks locally.
```bash title="SB CLI"
sb exec [--config-file]
        [--config-override]
```

#### Optional arguments

| Name | Default | Description |
| --- | --- | --- |
| `--config-file` `-c` | `None` | Path to SuperBench config file. |
| `--config-override` `-C` | `None` | Extra arguments to override config_file. |

#### Examples

Execute GPT2 model benchmark only in default benchmarking configuration:
```bash title="SB CLI"
sb exec --config-override superbench.enable="['gpt2_models']"
```

### `sb run`

Run the SuperBench benchmarks distributedly.
```bash title="SB CLI"
sb run [--config-file]
       [--config-override]
       [--docker-image]
       [--docker-password]
       [--docker-username]
       [--host-file]
       [--host-list]
       [--host-password]
       [--host-username]
       [--private-key]
```

#### Optional arguments

| Name | Default | Description |
| --- | --- | --- |
| `--config-file` `-c` | `None` | Path to SuperBench config file. |
| `--config-override` `-C` | `None` | Extra arguments to override config_file. |
| `--docker-image` `-i` | `superbench/superbench` | Docker image URI. |
| `--docker-password` | `None` | Docker registry password if authentication is needed. |
| `--docker-username` | `None` | Docker registry username if authentication is needed. |
| `--host-file` `-f` | `None` | Path to Ansible inventory host file. |
| `--host-list` `-l` | `None` | Comma separated host list. |
| `--host-password` | `None` | Host password or key passphase if needed. |
| `--host-username` | `None` | Host username if needed. |
| `--private-key` | `None` | Path to private key if needed. |

#### Examples

Run all benchmarks on all managed nodes in `./host.yaml` using image `superbench/cuda:11.1`
and default benchmarking configuration:
```bash title="SB CLI"
sb run --docker-image superbench/cuda:11.1 --host-file ./host.yaml
```

### `sb version`

Print the current SuperBench CLI version.
```bash title="SB CLI"
sb version
```

#### Examples

Print version:
```bash title="SB CLI"
sb version
```
