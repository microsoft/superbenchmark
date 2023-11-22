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
        - name: deploy image "superbench/cuda:11.1" to all nodes in ./host.ini
          text: {cli_name} deploy --docker-image superbench/cuda:11.1 --host-file ./host.ini
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
        - name: run all benchmarks on all nodes in ./host.ini using image "superbench/cuda:11.1"
            and default benchmarking configuration
          text: {cli_name} run --docker-image superbench/cuda:11.1 --host-file ./host.ini
        - name: run kernel launch benchmarks on host directly without using Docker
          text: >
            {cli_name} run --no-docker --host-list localhost
            --config-override superbench.enable=kernel-launch superbench.env.SB_MICRO_PATH=/path/to/superbenchmark
        - name: Collect system info on all nodes in ./host.ini" without running benchmarks
          text: {cli_name} run --get-info --host-file ./host.ini -C superbench.enable=none
        - name: Collect system info on all nodes in ./host.ini" while running benchmarks
          text: {cli_name} run --get-info --host-file ./host.ini
""".format(cli_name=CLI_NAME)

helps['benchmark'] = """
    type: group
    short-summary: Commands to manage benchmarks.
"""

helps['benchmark list'] = """
    type: command
    examples:
        - name: list all benchmarks
          text: {cli_name} benchmark list
        - name: list all benchmarks ending with "-bw"
          text: {cli_name} benchmark list --name [a-z]+-bw
""".format(cli_name=CLI_NAME)

helps['benchmark list-parameters'] = """
    type: command
    examples:
        - name: list parameters for all benchmarks
          text: {cli_name} benchmark list-parameters
        - name: list parameters for all benchmarks which starts with "pytorch-"
          text: {cli_name} benchmark list-parameters --name pytorch-[a-z]+
""".format(cli_name=CLI_NAME)

helps['node'] = """
    type: group
    short-summary: Get detailed information or configurations on the local node.
"""

helps['node info'] = """
    type: command
    short-summary: Get system info.
    examples:
        - name: get system info of the local node
          text: {cli_name} node info
""".format(cli_name=CLI_NAME)

helps['result'] = """
    type: group
    short-summary: Process or analyze the results of SuperBench benchmarks.
"""

helps['result diagnosis'] = """
    type: command
    short-summary: >
        Filter the defective machines automatically from benchmarking results
        according to rules defined in rule file.
    examples:
        - name: run data diagnosis and output the results in excel format
          text: >
            {cli_name} result diagnosis
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --baseline-file baseline.json
            --output-file-format excel
        - name: run data diagnosis and output the results in jsonl format
          text: >
            {cli_name} result diagnosis
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --baseline-file baseline.json
            --output-file-format jsonl
        - name: run data diagnosis and output the results in json format
          text: >
            {cli_name} result diagnosis
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --baseline-file baseline.json
            --output-file-format json
        - name: run data diagnosis and output the results in markdown format
          text: >
            {cli_name} result diagnosis
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --baseline-file baseline.json
            --output-file-format md
        - name: run data diagnosis and output the results in html format
          text: >
            {cli_name} result diagnosis
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --baseline-file baseline.json
            --output-file-format html
        - name: run data diagnosis and output the results of all nodes in json format
          text: >
            {cli_name} result diagnosis
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --baseline-file baseline.json
            --output-file-format json
            --output-all
""".format(cli_name=CLI_NAME)

helps['result summary'] = """
    type: command
    short-summary: >
        Generate the readable summary of benchmarking results
        according to rules defined in rule file.
    examples:
        - name: run result summary and output the results in excel format
          text: >
            {cli_name} result summary
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --output-file-format excel
        - name: run result summary and output the results in markdown format
          text: >
            {cli_name} result summary
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --output-file-format md
        - name: run result summary and output the results in html format
          text: >
            {cli_name} result summary
            --data-file outputs/results-summary.jsonl
            --rule-file rule.yaml
            --output-file-format html
""".format(cli_name=CLI_NAME)

helps['result generate-baseline'] = """
    type: command
    short-summary: >
        Generate the baseline of benchmarking results from jsonline file
        according to rules defined in rule file.
    examples:
        - name: run result generate-baseline to generate baseline.json file
          text: >
            {cli_name} result generate-baseline
            --data-file outputs/results-summary.jsonl
            --summary-rule-file summary-rule.yaml
            --diagnosis-rule-file diagnosis-rule.yaml
        - name: run result generate-baseline and merge with previous baseline
          text: >
            {cli_name} result generate-baseline
            --data-file outputs/results-summary.jsonl
            --summary-rule-file summary-rule.yaml
            --diagnosis-rule-file diagnosis-rule.yaml
            --baseline-file previous-baseline.json
""".format(cli_name=CLI_NAME)


class SuperBenchCLIHelp(CLIHelp):
    """SuperBench CLI help loader."""
    def __init__(self, cli_ctx=None):
        """Init CLI help loader.

        Args:
            cli_ctx (knack.cli.CLI, optional): CLI Context. Defaults to None.
        """
        super().__init__(cli_ctx=cli_ctx, welcome_message=WELCOME_MESSAGE)
