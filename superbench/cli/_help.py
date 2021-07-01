# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI help."""

from knack.help import CLIHelp
from knack.help_files import helps

CLI_NAME = 'sb'
WELCOME_MESSAGE = r"""
   _____                       ____                  _
  / ____|                     |  _ \                | |
 | (___  _   _ _ __   ___ _ __| |_) | ___ _ __   ___| |__
  \___ \| | | | '_ \ / _ \ '__|  _ < / _ \ '_ \ / __| '_ \
  ____) | |_| | |_) |  __/ |  | |_) |  __/ | | | (__| | | |
 |_____/ \__,_| .__/ \___|_|  |____/ \___|_| |_|\___|_| |_|
              | |
              |_|

Welcome to the SB CLI!
"""

helps['version'] = """
    type: command
    short-summary: Print the current SuperBench CLI version.
    examples:
        - name: print version
          text: {cli_name} version
""".format(cli_name=CLI_NAME)

helps['deploy'] = """
    type: command
    short-summary: Deploy the SuperBench environments to all given nodes.
    examples:
        - name: deploy default image on local GPU node
          text: {cli_name} deploy --host-list localhost
        - name: deploy image "superbench/cuda:11.1" to all nodes in ./host.yaml
          text: {cli_name} deploy --docker-image superbench/cuda:11.1 --host-file ./host.yaml
        - name: deploy image "superbench/rocm:4.0" to node-0 and node-2, using key file id_rsa for ssh
          text: {cli_name} deploy --docker-image superbench/rocm:4.0 --host-list node-0,node-2 --private-key id_rsa
""".format(cli_name=CLI_NAME)

helps['exec'] = """
    type: command
    short-summary: Execute the SuperBench benchmarks locally.
    examples:
        - name: execute all benchmarks using image "superbench/cuda:11.1" and default benchmarking configuration
          text: {cli_name} exec --docker-image superbench/cuda:11.1
        - name: execute all benchmarks using image "superbench/rocm:4.0" and custom config file ./config.yaml
          text: {cli_name} exec --docker-image superbench/rocm:4.0 --config-file ./config.yaml
""".format(cli_name=CLI_NAME)

helps['run'] = """
    type: command
    short-summary: Run the SuperBench benchmarks distributedly.
    examples:
        - name: run all benchmarks on local GPU node
          text: {cli_name} run --host-list localhost
        - name: run all benchmarks on all nodes in ./host.yaml using image "superbench/cuda:11.1"
            and default benchmarking configuration
          text: {cli_name} run --docker-image superbench/cuda:11.1 --host-file ./host.yaml
""".format(cli_name=CLI_NAME)


class SuperBenchCLIHelp(CLIHelp):
    """SuperBench CLI help loader."""
    def __init__(self, cli_ctx=None):
        """Init CLI help loader.

        Args:
            cli_ctx (knack.cli.CLI, optional): CLI Context. Defaults to None.
        """
        super().__init__(cli_ctx=cli_ctx, welcome_message=WELCOME_MESSAGE)
