# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Process Utility."""

import subprocess


def run_command(command):
    """Run command in string format, return the result with stdout and stderr.

    Args:
        command (str): command to run.

    Return:
        result (subprocess.CompletedProcess): The return value from subprocess.run().
    """
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=False, universal_newlines=True
    )

    return result
