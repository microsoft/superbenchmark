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

The following lists `sb` commands usages and examples:

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

#### Global arguments

| Name | Default | Description |
| --- | --- | --- |
| `--help` `-h` | N/A | Show help message. |

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

#### Global arguments

| Name | Default | Description |
| --- | --- | --- |
| `--help` `-h` | N/A | Show help message. |

#### Examples

Execute GPT2 model benchmark in default configuration:
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

#### Global arguments

| Name | Default | Description |
| --- | --- | --- |
| `--help` `-h` | N/A | Show help message. |

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

#### Global arguments

| Name | Default | Description |
| --- | --- | --- |
| `--help` `-h` | N/A | Show help message. |

#### Examples

Print version:
```bash title="SB CLI"
sb version
```
