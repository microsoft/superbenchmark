# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI command handler."""

import os
import yaml
from pathlib import Path

from knack.util import CLIError

import superbench
from superbench.common.utils import get_sb_command, get_config, new_output_dir


def check_argument_file(name, file):
    """Check file path in CLI arguments.

    Args:
        name (str): argument name.
        file (str): file path.

    Raises:
        CLIError: If file does not exist.
    """
    if file and not Path(file).exists():
        raise CLIError('{} {} does not exist.'.format(name, file))


def version_command_handler():
    """Print the current SuperBench tool version.

    Returns:
        str: current SuperBench tool version.
    """
    return superbench.__version__


def deploy_command_handler(
    docker_image,
    docker_username=None,
    docker_password=None,
    host_file=None,
    host_list=None,
    host_username=None,
    host_password=None,
    private_key=None
):
    """Deploy the SuperBench environments to all given nodes.

    Deploy SuperBench environments on all nodes, including:
    1. check drivers
    2. install required system dependencies
    3. install Docker and container runtime
    4. pull Docker image

    Args:
        docker_image (str): Docker image URI.
        docker_username (str, optional): Docker registry username if authentication is needed. Defaults to None.
        docker_password (str, optional): Docker registry password if authentication is needed. Defaults to None.
        host_file (str, optional): Path to Ansible inventory host file. Defaults to None.
        host_list (str, optional): Comma separated host list. Defaults to None.
        host_username (str, optional): Host username if needed. Defaults to None.
        host_password (str, optional): Host password or key passphase if needed. Defaults to None.
        private_key (str, optional): Path to private key if needed. Defaults to None.

    Raises:
        CLIError: If input arguments are invalid.
    """
    if not (host_file or host_list):
        raise CLIError('Must specify one of host_file or host_list.')
    check_argument_file('host_file', host_file)
    check_argument_file('private_key', private_key)

    raise NotImplementedError


def exec_command_handler(
    docker_image, docker_username=None, docker_password=None, config_file=None, config_override=None
):
    """Run the SuperBench benchmarks locally.

    Args:
        docker_image (str): Docker image URI.
        docker_username (str, optional): Docker registry username if authentication is needed. Defaults to None.
        docker_password (str, optional): Docker registry password if authentication is needed. Defaults to None.
        config_file (str, optional): Path to SuperBench config file. Defaults to None.
        config_override (str, optional): Extra arguments to override config_file,
            following [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic). Defaults to None.

    Raises:
        CLIError: If input arguments are invalid.
    """
    if bool(docker_username) != bool(docker_password):
        raise CLIError('Must specify both docker_username and docker_password if authentication is needed.')
    check_argument_file('config_file', config_file)

    # dump configs into outputs/date/merge.yaml
    config = get_config(config_file)
    config['docker'] = {}
    for n in ['image', 'username', 'password']:
        config['docker'][n] = eval('docker_{}'.format(n))
    config['ansible'] = {}
    for n in ['file', 'list', 'username', 'password']:
        config['ansible']['host_{}'.format(n)] = eval('host_{}'.format(n))
    output_dir = new_output_dir()
    with (Path(output_dir) / 'merge.yaml').open(mode='w') as f:
        yaml.safe_dump(config, f)
    os.system(get_sb_command('sb-exec', output_dir, config_override or ''))


def run_command_handler(
    docker_image,
    docker_username=None,
    docker_password=None,
    host_file=None,
    host_list=None,
    host_username=None,
    host_password=None,
    private_key=None,
    config_file=None,
    config_override=None
):
    """Run the SuperBench benchmarks distributedly.

    Run all benchmarks on given nodes.

    Args:
        docker_image (str): Docker image URI.
        docker_username (str, optional): Docker registry username if authentication is needed. Defaults to None.
        docker_password (str, optional): Docker registry password if authentication is needed. Defaults to None.
        host_file (str, optional): Path to Ansible inventory host file. Defaults to None.
        host_list (str, optional): Comma separated host list. Defaults to None.
        host_username (str, optional): Host username if needed. Defaults to None.
        host_password (str, optional): Host password or key passphase if needed. Defaults to None.
        private_key (str, optional): Path to private key if needed. Defaults to None.
        config_file (str, optional): Path to SuperBench config file. Defaults to None.
        config_override (str, optional): Extra arguments to override config_file,
            following [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic). Defaults to None.

    Raises:
        CLIError: If input arguments are invalid.
    """
    if bool(docker_username) != bool(docker_password):
        raise CLIError('Must specify both docker_username and docker_password if authentication is needed.')
    if not (host_file or host_list):
        raise CLIError('Must specify one of host_file or host_list.')
    check_argument_file('host_file', host_file)
    check_argument_file('private_key', private_key)
    check_argument_file('config_file', config_file)

    # dump configs into outputs/date/merge.yaml
    config = get_config(config_file)
    config['docker'] = {}
    for n in ['image', 'username', 'password']:
        config['docker'][n] = eval('docker_{}'.format(n))
    config['ansible'] = {}
    for n in ['file', 'list', 'username', 'password']:
        config['ansible']['host_{}'.format(n)] = eval('host_{}'.format(n))
    output_dir = new_output_dir()
    with (Path(output_dir) / 'merge.yaml').open(mode='w') as f:
        yaml.safe_dump(config, f)
    os.system(get_sb_command('sb-run', output_dir, config_override or ''))
