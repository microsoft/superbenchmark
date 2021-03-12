# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for command."""


def get_sb_command(cli, output_path, config_override):
    """Get sb command for sb-run or sb-exec.

    Args:
        cli (str): CLI name.
        output_path (str): Output directory path.
        config_override (str): Extra arguments to override config.

    Returns:
        str: Command to run.
    """
    sb_cmd = '{cli} ' \
        '--config-name=config.merge ' \
        '--config-dir={path} ' \
        'hydra.run.dir={path} ' \
        'hydra.sweep.dir={path} ' \
        '{args}'.format(cli=cli, path=output_path, args=config_override)
    return sb_cmd.strip()
