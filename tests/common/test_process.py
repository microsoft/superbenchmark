# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test common.util.process."""

from superbench.common.utils import run_command


def test_run_command():
    """Test run_command."""
    command = 'echo 123'
    expected_output = '123\n'
    output = run_command(command)
    assert (output.stdout == expected_output)
    assert (output.returncode == 0)
    output = run_command(command, flush_output=True)
    assert (output.stdout == expected_output)
    assert (output.returncode == 0)

    command = 'abb'
    output = run_command(command)
    assert (output.returncode != 0)
    output = run_command(command, flush_output=True)
    assert (output.returncode != 0)
