# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for developer-container initialization commands."""

from pathlib import Path
import shlex

from pip._internal.commands import create_command


def test_post_create_pip_arguments():
    """Parse the actual install arguments without installing any packages."""
    script = Path(__file__).resolve().parents[1] / '.devcontainer' / 'post-create-command.sh'
    command = next(line for line in script.read_text().splitlines() if line.startswith('pip install '))
    arguments = shlex.split(command)[2:]
    _, requirements = create_command('install', isolated=True).parse_args(arguments)
    assert requirements == ['.[develop,cpuworker]']
    assert '--use-feature=in-tree-build' not in arguments
