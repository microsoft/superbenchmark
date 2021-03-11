#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench command line interface."""

import sys

from knack import CLI

import superbench
from superbench.cli._help import CLI_NAME, SuperBenchCLIHelp
from superbench.cli._commands import SuperBenchCommandsLoader


class SuperBenchCLI(CLI):
    """The main driver for SuperBench CLI."""
    def get_cli_version(self):
        """Get the CLI version.

        Returns:
            str: CLI semantic version.
        """
        return superbench.__version__

    @classmethod
    def get_cli(cls):
        """Get CLI instance.

        Returns:
            SuperBenchCLI: An instance for SuperBench CLI.
        """
        return cls(
            cli_name=CLI_NAME,
            config_env_var_prefix=CLI_NAME,
            commands_loader_cls=SuperBenchCommandsLoader,
            help_cls=SuperBenchCLIHelp,
        )


def main():
    """The main function for CLI."""
    sb_cli = SuperBenchCLI.get_cli()
    exit_code = sb_cli.invoke(sys.argv[1:])
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
