"""Wrapper around subprocess to run commands."""
from __future__ import annotations

import platform
import re
import subprocess
from typing import TYPE_CHECKING, List, Union
from uuid import uuid4

if TYPE_CHECKING:
    import pexpect


def _lazy_import_pexpect() -> pexpect:
    """Import pexpect only when needed."""
    if platform.system() == "Windows":
        raise ValueError("Persistent bash processes are not yet supported on Windows.")
    try:
        import pexpect

    except ImportError:
        raise ImportError(
            "pexpect required for persistent bash processes."
            " To install, run `pip install pexpect`."
        )
    return pexpect

class BashResult:
    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def to_str(self) -> str:
        return f"returncode: {self.returncode}\nstdout: {self.stdout}\nstderr: {self.stderr}"

class BashProcess:
    """Executes bash commands and returns the output."""

    def __init__(
        self,
        strip_newlines: bool = False,
        return_err_output: bool = True,
        persistent: bool = False,
        verbose: bool=True,
    ):
        """Initialize with stripping newlines."""
        self.strip_newlines = strip_newlines
        self.return_err_output = return_err_output
        self.prompt = ""
        self.process = None
        self._verbose = verbose
        if persistent:
            self.prompt = str(uuid4())
            self.process = self._initialize_persistent_process(self.prompt)

    @staticmethod
    def _initialize_persistent_process(prompt: str) -> pexpect.spawn:
        # Start bash in a clean environment
        # Doesn't work on windows
        pexpect = _lazy_import_pexpect()
        process = pexpect.spawn(
            "env", ["-i", "bash", "--norc", "--noprofile"], encoding="utf-8"
        )
        # Set the custom prompt
        process.sendline("PS1=" + prompt)

        process.expect_exact(prompt, timeout=10)
        return process

    def run(self, commands: Union[str, List[str]]) -> str:
        """Run commands and return final output."""
        if isinstance(commands, str):
            commands = [commands]
        commands = ";".join(commands)
        if self.process is not None:
            return self._run_persistent(
                commands,
            )
        else:
            return self._run(commands)

    def _run(self, command: str) -> str:
        """Run commands and return final output."""
        bash_res = self._run_command(command)

        return bash_res.to_str() 
    
    def _run_command(self, command: str) -> BashResult:
        """Runs a Linux command and returns the output."""
        if self._verbose:
            print(f"\tRan command: \033[32m\033[1m{command}\033[0m :")

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = ""
        while process.poll() is None:
            line=process.stdout.readline().decode("utf8")
            stdout += line
            if self._verbose:
                print(f"\t\t\033[32m{line.strip()}\033[0m")
        err = process.stderr.readlines()
        stderr = "".join([line.decode("utf8") for line in err])
        perr = stderr.replace("\n", "\n\t\t")
        if self._verbose:
            print(f"\t\t\033[31m{perr}\033[0m")
        return_code = process.returncode

        if stdout !="":
            stdout = "\n".join(stdout.split("\n")[-200:])
        if stderr !="":
            stderr = {stderr[-200:]}

        bash_res = BashResult(stdout, stderr, return_code)
        return bash_res
    
    def process_output(self, output: str, command: str) -> str:
        # Remove the command from the output using a regular expression
        pattern = re.escape(command) + r"\s*\n"
        output = re.sub(pattern, "", output, count=1)
        return output.strip()

    def _run_persistent(self, command: str) -> str:
        """Run commands and return final output."""
        pexpect = _lazy_import_pexpect()
        if self.process is None:
            raise ValueError("Process not initialized")
        self.process.sendline(command)

        # Clear the output with an empty string
        self.process.expect(self.prompt, timeout=10)
        self.process.sendline("")

        try:
            self.process.expect([self.prompt, pexpect.EOF], timeout=10)
        except pexpect.TIMEOUT:
            return f"Timeout error while executing command {command}"
        if self.process.after == pexpect.EOF:
            return f"Exited with error status: {self.process.exitstatus}"
        output = self.process.before
        output = self.process_output(output, command)
        if self.strip_newlines:
            return output.strip()
        return output
