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

### `sb benchmark list`

List benchmarks which match the regular expression.
```bash title="SB CLI"
sb benchmark list [--name]
```

#### Optional arguments

| Name          | Default | Description                           |
|---------------|---------|---------------------------------------|
| `--name` `-n` | `None`  | Benchmark name or regular expression. |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

List all benchmarks:
```bash title="SB CLI"
sb benchmark list
```

List all benchmarks ending with "-bw":
```bash title="SB CLI"
sb benchmark list --name [a-z]+-bw
```

### `sb benchmark list-parameters`

List parameters for benchmarks which match the regular expression.
```bash title="SB CLI"
sb benchmark list-parameters [--name]
```

#### Optional arguments

| Name          | Default | Description                           |
|---------------|---------|---------------------------------------|
| `--name` `-n` | `None`  | Benchmark name or regular expression. |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

List parameters for all benchmarks:
```bash title="SB CLI"
sb benchmark list-parameters
```

List parameters for all benchmarks which starts with "pytorch-":
```bash title="SB CLI"
sb benchmark list-parameters --name pytorch-[a-z]+
```

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
          [--no-image-pull]
          [--output-dir]
          [--private-key]
```

#### Optional arguments

| Name                  | Default                 | Description                                                                       |
|-----------------------|-------------------------|-----------------------------------------------------------------------------------|
| `--docker-image` `-i` | `superbench/superbench` | Docker image URI, [here](./user-tutorial/container-images.mdx) listed all images. |
| `--docker-password`   | `None`                  | Docker registry password if authentication is needed.                             |
| `--docker-username`   | `None`                  | Docker registry username if authentication is needed.                             |
| `--host-file` `-f`    | `None`                  | Path to Ansible inventory host file.                                              |
| `--host-list` `-l`    | `None`                  | Comma separated host list.                                                        |
| `--host-password`     | `None`                  | Host password or key passphase if needed.                                         |
| `--host-username`     | `None`                  | Host username if needed.                                                          |
| `--no-image-pull`     | `False`                 | Skip pull and use local Docker image.                                             |
| `--output-dir`        | `None`                  | Path to output directory, outputs/{datetime} will be used if not specified.       |
| `--private-key`       | `None`                  | Path to private key if needed.                                                    |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

Deploy default image on local GPU node:
```bash title="SB CLI"
sb deploy --host-list localhost
```

Deploy image `superbench/cuda:11.1` to all nodes in `./host.ini`:
```bash title="SB CLI"
sb deploy --docker-image superbench/cuda:11.1 --host-file ./host.ini
```

### `sb exec`

Execute the SuperBench benchmarks locally.
```bash title="SB CLI"
sb exec [--config-file]
        [--config-override]
        [--output-dir]
```

#### Optional arguments

| Name                     | Default | Description                                                                 |
|--------------------------|---------|-----------------------------------------------------------------------------|
| `--config-file` `-c`     | `None`  | Path to SuperBench config file.                                             |
| `--config-override` `-C` | `None`  | Extra arguments to override config_file.                                    |
| `--output-dir`           | `None`  | Path to output directory, outputs/{datetime} will be used if not specified. |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

Execute GPT2 model benchmark in default configuration:
```bash title="SB CLI"
sb exec --config-override superbench.enable="['gpt2_models']"
```

### `sb node info`
Get system info on the local node.

```bash title="SB CLI"
sb node info [--output-dir]
```

#### Optional arguments

| Name           | Default | Description                                                                 |
|----------------|---------|-----------------------------------------------------------------------------|
| `--output-dir` | `None`  | Path to output directory, outputs/{datetime} will be used if not specified. |

#### Examples

Get system info on the local node and save it into the `outputs` dir:
```bash title="SB CLI"
sb node info --output-dir outputs
```

### `sb result diagnosis`

Filter the defective machines automatically from benchmarking results according to rules defined in rule file.

```bash title="SB CLI"
sb result diagnosis --baseline-file
                    --data-file
                    --rule-file
                    [--decimal-place-value]
                    [--rule-file]
                    [--output-all]
                    [--output-dir]
                    [--output-file-format {excel, json, md, html}]
```

#### Required arguments

| Name               | Description            |
|--------------------|------------------------|
| `--data-file` `-d` | Path to raw data file. |
| `--rule-file` `-r` | Path to rule file.     |

#### Optional arguments

| Name                    | Default | Description                                                                 |
|-------------------------|---------|-----------------------------------------------------------------------------|
| `--baseline-file` `-b` | Path to baseline file. |
| `--decimal-place-value` | 2       | Number of valid decimal places to show in output. Default: 2.               |
| `--output-all`          | N/A     | Output diagnosis results for all nodes.                                     |
| `--output-dir`          | `None`  | Path to output directory, outputs/{datetime} will be used if not specified. |
| `--output-file-format`  | `excel` | Format of output file, 'excel', 'json', 'jsonl', 'md' or 'html'. Default: excel.     |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

Run data diagnosis and output the results in excel format:
```bash title="SB CLI"
sb result diagnosis --data-file outputs/results-summary.jsonl --rule-file rule.yaml --baseline-file baseline.json --output-file-format excel
```

Run data diagnosis and output the results in json format:
```bash title="SB CLI"
sb result diagnosis --data-file outputs/results-summary.jsonl --rule-file rule.yaml --baseline-file baseline.json --output-file-format json
```

Run data diagnosis and output the results in jsonl format:
```bash title="SB CLI"
sb result diagnosis --data-file outputs/results-summary.jsonl --rule-file rule.yaml --baseline-file baseline.json --output-file-format jsonl
```

Run data diagnosis and output the results in markdown format with 2 valid decimal places:
```bash title="SB CLI"
sb result diagnosis --data-file outputs/results-summary.jsonl --rule-file rule.yaml --baseline-file baseline.json --output-file-format md --decimal-place-value 2
```

run data diagnosis and output the results of all nodes in json format:
```bash title="SB CLI"
sb result diagnosis --data-file outputs/results-summary.jsonl --rule-file rule.yaml --baseline-file baseline.json --output-file-format json --output-all
```

### `sb result summary`

Generate the readable summary report automatically from benchmarking results according to rules defined in rule file.

```bash title="SB CLI"
sb result summary --data-file
                  --rule-file
                  [--decimal-place-value]
                  [--output-dir]
                  [--output-file-format {md, excel, html}]
```

#### Required arguments

| Name               | Description            |
|--------------------|------------------------|
| `--data-file` `-d` | Path to raw data file. |
| `--rule-file` `-r` | Path to rule file.     |

#### Optional arguments

| Name                    | Default | Description                                                                 |
|-------------------------|---------|-----------------------------------------------------------------------------|
| `--decimal-place-value` | 2       | Number of valid decimal places to show in output. Default: 2.               |
| `--output-dir`          | `None`  | Path to output directory, outputs/{datetime} will be used if not specified. |
| `--output-file-format`  | `md`    | Format of output file, 'excel', 'md' or 'html'. Default: md.                |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

Run result summary and output the results in markdown format with 2 valid decimal places:
```bash title="SB CLI"
sb result summary --data-file outputs/results-summary.jsonl --rule-file rule.yaml --output-file-format md --decimal-place-value 2
```

Run result summary and output the results in html format:
```bash title="SB CLI"
sb result summary --data-file outputs/results-summary.jsonl --rule-file rule.yaml --output-file-format html
```

### `sb result generate-baseline`

Generate the baseline file automatically from multiple machines results according to rules defined in rule file.

```bash title="SB CLI"
sb result generate-baseline --data-file
                            --summary-rule-file
                            [--diagnosis-rule-file]
                            [--baseline-file]
                            [--decimal-place-value]
                            [--output-dir]
```

#### Required arguments

| Name                        | Description                             |
|-----------------------------|-----------------------------------------|
| `--data-file` `-d`          | Path to raw data file.                  |
| `--summary-rule-file` `-sr` | Path to summary rule file.              |

#### Optional arguments

| Name                          | Default | Description                                                                 |
|-------------------------------|---------|-----------------------------------------------------------------------------|
| `--diagnosis-rule-file` `-dr` | `None`  | Path to diagnosis rule file. Default: None.                                 |
| `--baseline-file` `-b`        | `None`  | Path to previous baseline file. Default: None.                              |
| `--decimal-place-value`       | 2       | Number of valid decimal places to show in output. Default: 2.               |
| `--output-dir`                | `None`  | Path to output directory, outputs/{datetime} will be used if not specified. |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

Run result generate-baseline to generate baseline.json file:
```bash title="SB CLI"
sb result generate-baseline --data-file outputs/results-summary.jsonl --summary-rule-file summary-rule.yaml --diagnosis-rule-file diagnosis-rule.yaml
```

Run result generate-baseline and merge with previous baseline:
```bash title="SB CLI"
sb result generate-baseline --data-file outputs/results-summary.jsonl --summary-rule-file summary-rule.yaml --diagnosis-rule-file diagnosis-rule.yaml --baseline-file previous-baseline.json
```

### `sb run`

Run the SuperBench benchmarks distributedly.
```bash title="SB CLI"
sb run [--config-file]
       [--config-override]
       [--docker-image]
       [--docker-password]
       [--docker-username]
       [--get-info]
       [--host-file]
       [--host-list]
       [--host-password]
       [--host-username]
       [--no-docker]
       [--output-dir]
       [--private-key]
```

#### Optional arguments

| Name                     | Default                 | Description                                                                 |
|--------------------------|-------------------------|-----------------------------------------------------------------------------|
| `--config-file` `-c`     | `None`                  | Path to SuperBench config file.                                             |
| `--config-override` `-C` | `None`                  | Extra arguments to override config_file.                                    |
| `--docker-image` `-i`    | `superbench/superbench` | Docker image URI.                                                           |
| `--docker-password`      | `None`                  | Docker registry password if authentication is needed.                       |
| `--docker-username`      | `None`                  | Docker registry username if authentication is needed.                       |
| `--get-info`             | `False`                 | Collect system info.                                                        |
| `--host-file` `-f`       | `None`                  | Path to Ansible inventory host file.                                        |
| `--host-list` `-l`       | `None`                  | Comma separated host list.                                                  |
| `--host-password`        | `None`                  | Host password or key passphase if needed.                                   |
| `--host-username`        | `None`                  | Host username if needed.                                                    |
| `--no-docker`            | `False`                 | Run on host directly without Docker.                                        |
| `--output-dir`           | `None`                  | Path to output directory, outputs/{datetime} will be used if not specified. |
| `--private-key`          | `None`                  | Path to private key if needed.                                              |

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

Run all benchmarks on local GPU node:
```bash title="SB CLI"
sb run --host-list localhost
```

Run all benchmarks on all managed nodes in `./host.ini` using image `superbench/cuda:11.1`
and default benchmarking configuration:
```bash title="SB CLI"
sb run --docker-image superbench/cuda:11.1 --host-file ./host.ini
```

Run kernel launch benchmarks on host directly without using Docker:
```bash title="SB CLI"
sb run --no-docker --host-list localhost --config-override \
  superbench.enable=kernel-launch superbench.env.SB_MICRO_PATH=/path/to/superbenchmark
```

Collect system info on all nodes in ./host.ini" distributed without running benchmarks:
```bash title="SB CLI"
sb run --get-info --host-file ./host.ini -C superbench.enable=none
```

Collect system info on all nodes in ./host.ini" distributed while running benchmarks:
```bash title="SB CLI"
sb run --get-info --host-file ./host.ini
```

### `sb version`

Print the current SuperBench CLI version.
```bash title="SB CLI"
sb version
```

#### Global arguments

| Name          | Default | Description        |
|---------------|---------|--------------------|
| `--help` `-h` | N/A     | Show help message. |

#### Examples

Print version:
```bash title="SB CLI"
sb version
```
