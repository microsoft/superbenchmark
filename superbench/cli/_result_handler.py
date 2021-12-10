# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI result subgroup command handler."""

from knack.util import CLIError

from superbench.analyzer import DataDiagnosis
from superbench.common.utils import create_sb_output_dir
from superbench.cli._handler import check_argument_file


def diagnosis_command_handler(raw_data_file, rule_file, baseline_file, output_dir=None, output_format='excel'):
    """Run data diagnosis.

    Args:
        raw_data_file (str): the path of raw data jsonl file.
        rule_file (str): The path of baseline yaml file
        baseline_file (str): The path of baseline json file
        output_dir (str): the directory of output file
        output_format (str): the format of the output, 'excel' or 'json'. Defaults to 'excel'.
    """
    try:
        # Create output directory
        sb_output_dir = create_sb_output_dir(output_dir)
        # Check arguments
        if output_format not in ['excel', 'json']:
            raise CLIError('Output format must be excel or json.')
        check_argument_file('raw_data_file', raw_data_file)
        check_argument_file('rule_file', rule_file)
        check_argument_file('baseline_file', baseline_file)
        # Run data diagnosis
        DataDiagnosis().run(raw_data_file, rule_file, baseline_file, sb_output_dir, output_format)
    except Exception as ex:
        raise RuntimeError('Failed to run diagnosis command.') from ex
