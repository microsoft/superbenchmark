# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Process Utility."""

import subprocess
import os
import shlex

from superbench.common.utils import stdout_logger


def run_command(command, quite=False, flush_output=False):
    """Run command in string format, return the result with stdout and stderr.
    Args:
        command (str): command to run.
        quite (bool): no stdout display of the command if quite is True. 
        flush_output (bool): enable real-time output flush or not when running the command.
    Return:
        result (subprocess.CompletedProcess): The return value from subprocess.run().
    """
    if flush_output:
        process = None
        try:
            args = shlex.split(command)
            process = subprocess.Popen(
                args, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
            )
            output = ''
            for line in process.stdout:
                output += line
                if not quite:
                    stdout_logger.log(line)
            process.wait()
            retcode = process.poll()
            return subprocess.CompletedProcess(args=args, returncode=retcode, stdout=output, stderr=output)
        except Exception as e:
            if process:
                process.kill()
                process.wait()
            return subprocess.CompletedProcess(args=args, returncode=-1, stdout=str(e), stderr=str(e))
    else:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=False, universal_newlines=True
        )
        if not quite:
            stdout_logger.log(result)
        return result
