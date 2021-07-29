---
id: installation
---

# Installation

SuperBench is used to run validations for AI infrastructure,
thus you need to prepare one __control node__ which is used to run SuperBench commands,
and one or multiple __managed nodes__ which are going to be validated.

Usually __control node__ could be a CPU node, while __managed nodes__ are GPU nodes with high speed inter-connection.

:::tip Tips
It is fine if you have only one GPU node and want to try SuperBench on it.
Control node and managed node can co-locate on the same machine.
:::

## Control node

Here're the system requirements for control node.

### Requirements

* Latest version of Linux, you're highly encouraged to use Ubuntu 18.04 or later.
* [Python](https://www.python.org/) version 3.6 or later (which can be checked by running `python3 --version`).
* [Pip](https://pip.pypa.io/en/stable/installing/) version 18.0 or later (which can be checked by running `python3 -m pip --version`).

:::note
Windows is not supported due to lack of Ansible support, but you still can use WSL2.
:::

Besides, control node should be able to access all managed nodes through SSH.
If you are going to use password instead of private key for SSH, you also need to install `sshpass`.

```bash
sudo apt-get install sshpass
```

It is also recommended to use [venv](https://docs.python.org/3/library/venv.html) for virtual environments,
but it is not strictly necessary.

```bash
# create a new virtual environment
python3 -m venv --system-site-packages ./venv
# activate the virtual environment
source ./venv/bin/activate

# exit the virtual environment later
# after you finish running superbench
deactivate
```

### Build

You can clone the source from GitHub and build it.

:::note Note
You should checkout corresponding tag to use release version, for example,

`git clone -b v0.2.1 https://github.com/microsoft/superbenchmark`
:::

```bash
git clone https://github.com/microsoft/superbenchmark
cd superbenchmark

python3 -m pip install .
make postinstall
```

After installation, you should be able to run SB CLI.

```bash
sb
```

## Managed nodes

Here're the system requirements for all managed GPU nodes.

### Requirements

* Latest version of Linux, you're highly encouraged to use Ubuntu 18.04 or later.
* Compatible GPU drivers should be install correctly.
  * For NVIDIA GPUs, driver version can be checked by running `nvidia-smi`.
* [Docker CE](https://docs.docker.com/engine/install/) version 19.03 or later (which can be checked by running `docker --version`).
* GPU support in Docker.
  * For NVIDIA GPUs, install
  [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
