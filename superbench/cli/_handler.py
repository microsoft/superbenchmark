# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI command handler."""

import sys
from pathlib import Path
from importlib_metadata import version, PackageNotFoundError

from knack.util import CLIError
from omegaconf import OmegaConf

import superbench
from superbench.runner import SuperBenchRunner
from superbench.executor import SuperBenchExecutor
from superbench.common.utils import create_sb_output_dir, get_sb_config


def check_argument_file(name, file):
    """Check file path in CLI arguments.

    Args:
        name (str): argument name.
        file (str): file path.

    Returns:
        str: Absolute file path if it exists.

    Raises:
        CLIError: If file does not exist.
    """
    if file:
        if not Path(file).exists():
            raise CLIError('{} {} does not exist.'.format(name, file))
        return str(Path(file).resolve())
    return file


def split_docker_domain(name):
    """Split Docker image name to domain and remainder part.

    Ported from https://github.com/distribution/distribution/blob/v2.7.1/reference/normalize.go#L62-L76.

    Args:
        name (str): Docker image name.

    Returns:
        str: Docker registry domain.
        str: Remainder part.
    """
    legacy_default_domain = 'index.docker.io'
    default_domain = 'docker.io'

    i = name.find('/')
    domain, remainder = '', ''
    if i == -1 or ('.' not in name[:i] and ':' not in name[:i] and name[:i] != 'localhost'):
        domain, remainder = default_domain, name
    else:
        domain, remainder = name[:i], name[i + 1:]
    if domain == legacy_default_domain:
        domain = default_domain
    if domain == default_domain and '/' not in remainder:
        remainder = 'library/{}'.format(remainder)
    return domain, remainder


def process_config_arguments(config_file=None, config_override=None, output_dir=None):
    """Process configuration arguments.

    Args:
        config_file (str, optional): Path to SuperBench config file. Defaults to None.
        config_override (str, optional): Extra arguments to override config_file,
            following [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic). Defaults to None.
        output_dir (str, optional): Path to output directory. Defaults to None.

    Returns:
        DictConfig: SuperBench config object.
        str: Dir for output.

    Raises:
        CLIError: If input arguments are invalid.
    """
    config_file = check_argument_file('config_file', config_file)

    # SuperBench config
    sb_config = get_sb_config(config_file)
    if config_override:
        sb_config_from_override = OmegaConf.from_dotlist(config_override)
        sb_config = OmegaConf.merge(sb_config, sb_config_from_override)

    # Create output directory
    sb_output_dir = create_sb_output_dir(output_dir)

    return sb_config, sb_output_dir


def process_runner_arguments(
    docker_image='superbench/superbench',
    docker_username=None,
    docker_password=None,
    no_docker=False,
    no_image_pull=False,
    host_file=None,
    host_list=None,
    host_username=None,
    host_password=None,
    private_key=None,
    output_dir=None,
    config_file=None,
    config_override=None
):
    """Process runner related arguments.

    Args:
        docker_image (str, optional): Docker image URI. Defaults to superbench/superbench:latest.
        docker_username (str, optional): Docker registry username if authentication is needed. Defaults to None.
        docker_password (str, optional): Docker registry password if authentication is needed. Defaults to None.
        no_docker (bool, optional): Run on host directly without Docker. Defaults to False.
        no_image_pull (bool, optional): Skip pull and use local Docker image. Defaults to False.
        host_file (str, optional): Path to Ansible inventory host file. Defaults to None.
        host_list (str, optional): Comma separated host list. Defaults to None.
        host_username (str, optional): Host username if needed. Defaults to None.
        host_password (str, optional): Host password or key passphase if needed. Defaults to None.
        private_key (str, optional): Path to private key if needed. Defaults to None.
        output_dir (str, optional): Path to output directory. Defaults to None.
        config_file (str, optional): Path to SuperBench config file. Defaults to None.
        config_override (str, optional): Extra arguments to override config_file,
            following [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic). Defaults to None.

    Returns:
        DictConfig: SuperBench config object.
        DictConfig: Docker config object.
        DictConfig: Ansible config object.
        str: Dir for output.

    Raises:
        CLIError: If input arguments are invalid.
    """
    if bool(docker_username) != bool(docker_password):
        raise CLIError('Must specify both docker_username and docker_password if authentication is needed.')
    if not (host_file or host_list):
        raise CLIError('Must specify one of host_file or host_list.')
    host_file = check_argument_file('host_file', host_file)
    private_key = check_argument_file('private_key', private_key)

    # Docker config
    docker_config = OmegaConf.create(
        {
            'image': docker_image,
            'username': docker_username,
            'password': docker_password,
            'registry': split_docker_domain(docker_image)[0],
            'skip': no_docker,
            'pull': not no_image_pull,
        }
    )
    # Ansible config
    ansible_config = OmegaConf.create(
        {
            'host_file': host_file,
            'host_list': host_list,
            'host_username': host_username,
            'host_password': host_password,
            'private_key': private_key,
        }
    )

    sb_config, sb_output_dir = process_config_arguments(
        config_file=config_file,
        config_override=config_override,
        output_dir=output_dir,
    )

    return docker_config, ansible_config, sb_config, sb_output_dir


def version_command_handler():
    """Print the current SuperBench tool version in "{last tag}+g{git hash}.d{date}" format.

    Returns:
        str: current SuperBench tool version.
    """
    try:
        return version('superbench')
    except PackageNotFoundError:
        return superbench.__version__


def exec_command_handler(config_file=None, config_override=None, output_dir=None):
    """Run the SuperBench benchmarks locally.

    Args:
        config_file (str, optional): Path to SuperBench config file. Defaults to None.
        config_override (str, optional): Extra arguments to override config_file,
            following [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic). Defaults to None.
        output_dir (str, optional): Path to output directory. Defaults to None.

    Raises:
        CLIError: If input arguments are invalid.
    """
    sb_config, sb_output_dir = process_config_arguments(
        config_file=config_file,
        config_override=config_override,
        output_dir=output_dir,
    )

    executor = SuperBenchExecutor(sb_config, sb_output_dir)
    executor.exec()


def deploy_command_handler(
    docker_image='superbench/superbench',
    docker_username=None,
    docker_password=None,
    no_image_pull=False,
    host_file=None,
    host_list=None,
    host_username=None,
    host_password=None,
    output_dir=None,
    private_key=None
):
    """Deploy the SuperBench environments to all given nodes.

    Deploy SuperBench environments on all nodes, including:
    1. check drivers
    2. install required system dependencies
    3. install Docker and container runtime
    4. pull Docker image

    Args:
        docker_image (str, optional): Docker image URI. Defaults to superbench/superbench:latest.
        docker_username (str, optional): Docker registry username if authentication is needed. Defaults to None.
        docker_password (str, optional): Docker registry password if authentication is needed. Defaults to None.
        no_image_pull (bool, optional): Skip pull and use local Docker image. Defaults to False.
        host_file (str, optional): Path to Ansible inventory host file. Defaults to None.
        host_list (str, optional): Comma separated host list. Defaults to None.
        host_username (str, optional): Host username if needed. Defaults to None.
        host_password (str, optional): Host password or key passphase if needed. Defaults to None.
        output_dir (str, optional): Path to output directory. Defaults to None.
        private_key (str, optional): Path to private key if needed. Defaults to None.

    Raises:
        CLIError: If input arguments are invalid.
    """
    docker_config, ansible_config, sb_config, sb_output_dir = process_runner_arguments(
        docker_image=docker_image,
        docker_username=docker_username,
        docker_password=docker_password,
        no_docker=False,
        no_image_pull=no_image_pull,
        host_file=host_file,
        host_list=host_list,
        host_username=host_username,
        host_password=host_password,
        output_dir=output_dir,
        private_key=private_key,
    )

    runner = SuperBenchRunner(sb_config, docker_config, ansible_config, sb_output_dir)
    runner.deploy()
    if runner.get_failure_count() != 0:
        sys.exit(runner.get_failure_count())


def run_command_handler(
    docker_image='superbench/superbench',
    docker_username=None,
    docker_password=None,
    no_docker=False,
    host_file=None,
    host_list=None,
    host_username=None,
    host_password=None,
    output_dir=None,
    private_key=None,
    config_file=None,
    config_override=None,
    get_info=False,
):
    """Run the SuperBench benchmarks distributedly.

    Run all benchmarks on given nodes.

    Args:
        docker_image (str, optional): Docker image URI. Defaults to superbench/superbench:latest.
        docker_username (str, optional): Docker registry username if authentication is needed. Defaults to None.
        docker_password (str, optional): Docker registry password if authentication is needed. Defaults to None.
        no_docker (bool, optional): Run on host directly without Docker. Defaults to False.
        host_file (str, optional): Path to Ansible inventory host file. Defaults to None.
        host_list (str, optional): Comma separated host list. Defaults to None.
        host_username (str, optional): Host username if needed. Defaults to None.
        host_password (str, optional): Host password or key passphase if needed. Defaults to None.
        output_dir (str, optional): Path to output directory. Defaults to None.
        private_key (str, optional): Path to private key if needed. Defaults to None.
        config_file (str, optional): Path to SuperBench config file. Defaults to None.
        config_override (str, optional): Extra arguments to override config_file,
            following [Hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic). Defaults to None.
        get_info (bool, optional): Collect node system info. Defaults to False.

    Raises:
        CLIError: If input arguments are invalid.
    """
    docker_config, ansible_config, sb_config, sb_output_dir = process_runner_arguments(
        docker_image=docker_image,
        docker_username=docker_username,
        docker_password=docker_password,
        no_docker=no_docker,
        no_image_pull=False,
        host_file=host_file,
        host_list=host_list,
        host_username=host_username,
        host_password=host_password,
        output_dir=output_dir,
        private_key=private_key,
        config_file=config_file,
        config_override=config_override,
    )

    runner = SuperBenchRunner(sb_config, docker_config, ansible_config, sb_output_dir)

    runner.run()
    if get_info:
        runner.run_sys_info()

    if runner.get_failure_count() != 0:
        sys.exit(runner.get_failure_count())
