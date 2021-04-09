# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI command handler."""

from pathlib import Path

from knack.util import CLIError
from omegaconf import OmegaConf

import superbench
from superbench.runner import SuperBenchRunner
from superbench.executor import SuperBenchExecutor
from superbench.common.utils import create_output_dir, get_sb_config


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
    if bool(docker_username) != bool(docker_password):
        raise CLIError('Must specify both docker_username and docker_password if authentication is needed.')
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

    # Docker config
    docker_config = OmegaConf.create()
    for key in ['image', 'username', 'password']:
        docker_config[key] = eval('docker_{}'.format(key))
    # SuperBench config
    sb_config = get_sb_config(config_file)
    if config_override:
        sb_config_from_override = OmegaConf.from_dotlist(config_override)
        sb_config = OmegaConf.merge(sb_config, sb_config_from_override)

    # Create output directory
    output_dir = create_output_dir()

    executor = SuperBenchExecutor(sb_config, docker_config, output_dir)
    executor.exec()


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

    # Docker config
    docker_config = OmegaConf.create()
    for key in ['image', 'username', 'password']:
        docker_config[key] = eval('docker_{}'.format(key))
    # Ansible config
    ansible_config = OmegaConf.create()
    for key in ['file', 'list', 'username', 'password']:
        ansible_config['host_{}'.format(key)] = eval('host_{}'.format(key))
    # SuperBench config
    sb_config = get_sb_config(config_file)
    if config_override:
        sb_config_from_override = OmegaConf.from_dotlist(config_override)
        sb_config = OmegaConf.merge(sb_config, sb_config_from_override)

    # Create output directory
    output_dir = create_output_dir()

    runner = SuperBenchRunner(sb_config, docker_config, ansible_config, output_dir)
    runner.run()
