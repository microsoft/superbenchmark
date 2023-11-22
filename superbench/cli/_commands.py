# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI commands."""

from knack.arguments import ArgumentsContext
from knack.commands import CLICommandsLoader, CommandGroup


class SuperBenchCommandsLoader(CLICommandsLoader):
    """SuperBench CLI commands loader."""
    def load_command_table(self, args):
        """Load commands into the command table.

        Args:
            args (list): List of arguments from the command line.

        Returns:
            collections.OrderedDict: Load commands into the command table.
        """
        with CommandGroup(self, '', 'superbench.cli._handler#{}') as g:
            g.command('version', 'version_command_handler')
            g.command('deploy', 'deploy_command_handler')
            g.command('exec', 'exec_command_handler')
            g.command('run', 'run_command_handler')
        with CommandGroup(self, 'benchmark', 'superbench.cli._benchmark_handler#{}') as g:
            g.command('list', 'benchmark_list_command_handler')
            g.command('list-parameters', 'benchmark_list_params_command_handler')
        with CommandGroup(self, 'node', 'superbench.cli._node_handler#{}') as g:
            g.command('info', 'info_command_handler')
        with CommandGroup(self, 'result', 'superbench.cli._result_handler#{}') as g:
            g.command('diagnosis', 'diagnosis_command_handler')
            g.command('summary', 'summary_command_handler')
            g.command('generate-baseline', 'generate_baseline_command_handler')
        return super().load_command_table(args)

    def load_arguments(self, command):
        """Load arguments for commands.

        Args:
            command: The command to load arguments for.
        """
        with ArgumentsContext(self, '') as ac:
            ac.argument('docker_image', options_list=('--docker-image', '-i'), type=str, help='Docker image URI.')
            ac.argument('docker_username', type=str, help='Docker registry username if authentication is needed.')
            ac.argument('docker_password', type=str, help='Docker registry password if authentication is needed.')
            ac.argument('no_docker', action='store_true', help='Run on host directly without Docker.')
            ac.argument('no_image_pull', action='store_true', help='Skip pull and use local Docker image.')
            ac.argument(
                'host_file', options_list=('--host-file', '-f'), type=str, help='Path to Ansible inventory host file.'
            )
            ac.argument('host_list', options_list=('--host-list', '-l'), type=str, help='Comma separated host list.')
            ac.argument('host_username', type=str, help='Host username if needed.')
            ac.argument('host_password', type=str, help='Host password or key passphase if needed.')
            ac.argument(
                'output_dir',
                type=str,
                help='Path to output directory, outputs/{datetime} will be used if not specified.'
            )
            ac.argument('private_key', type=str, help='Path to private key if needed.')
            ac.argument(
                'config_file', options_list=('--config-file', '-c'), type=str, help='Path to SuperBench config file.'
            )
            ac.argument(
                'config_override',
                options_list=('--config-override', '-C'),
                type=str,
                nargs='+',
                help='Extra arguments to override config_file.'
            )
            ac.argument(
                'get_info', options_list=('--get-info', '-g'), action='store_true', help='Collect node system info.'
            )

        with ArgumentsContext(self, 'benchmark') as ac:
            ac.argument('name', options_list=('--name', '-n'), type=str, help='Benchmark name or regular expression.')

        with ArgumentsContext(self, 'result') as ac:
            ac.argument('raw_data_file', options_list=('--data-file', '-d'), type=str, help='Path to raw data file.')
            ac.argument('rule_file', options_list=('--rule-file', '-r'), type=str, help='Path to rule file.')
            ac.argument(
                'summary_rule_file',
                options_list=('--summary-rule-file', '-sr'),
                type=str,
                help='Path to summary rule file.'
            )
            ac.argument(
                'diagnosis_rule_file',
                options_list=('--diagnosis-rule-file', '-dr'),
                type=str,
                help='Path to diagnosis rule file.'
            )
            ac.argument(
                'baseline_file', options_list=('--baseline-file', '-b'), type=str, help='Path to baseline file.'
            )
            ac.argument(
                'output_dir',
                type=str,
                help='Path to output directory, outputs/{datetime} will be used if not specified.'
            )
            ac.argument('output_file_format', type=str, help='Format of output file, excel or json.')
            ac.argument('decimal_place_value', type=int, help='Number of decimal places to show in output.')
            ac.argument('output_all', action='store_true', help='Output results of all nodes.')

        super().load_arguments(command)
