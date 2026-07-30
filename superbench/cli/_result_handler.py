# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI result subgroup command handler."""

from knack.util import CLIError

from superbench.analyzer import DataDiagnosis
from superbench.analyzer import ResultSummary
from superbench.analyzer import BaselineGeneration
from superbench.common.utils import create_sb_output_dir
from superbench.cli._handler import check_argument_file


def diagnosis_command_handler(
    raw_data_file,
    rule_file,
    baseline_file=None,
    output_dir=None,
    output_file_format='excel',
    output_all=False,
    decimal_place_value=2
):
    """Run data diagnosis.

    Args:
        raw_data_file (str): Path to raw data jsonl file.
        rule_file (str): Path to baseline yaml file.
        baseline_file (str): Path to baseline json file.
        output_dir (str): Path to output directory.
        output_file_format (str): Format of the output file, 'excel', 'json', 'md' or 'html'. Defaults to 'excel'.
        output_all (bool): output diagnosis results for all nodes
        decimal_place_value (int): Number of decimal places to show in output.
    """
    try:
        # Create output directory
        sb_output_dir = create_sb_output_dir(output_dir)
        # Check arguments
        supported_output_format = ['excel', 'json', 'md', 'html', 'jsonl']
        if output_file_format not in supported_output_format:
            raise CLIError('Output format must be in {}.'.format(str(supported_output_format)))
        check_argument_file('raw_data_file', raw_data_file)
        check_argument_file('rule_file', rule_file)
        if baseline_file:
            check_argument_file('baseline_file', baseline_file)
        # Run data diagnosis
        DataDiagnosis().run(
            raw_data_file, rule_file, baseline_file, sb_output_dir, output_file_format, output_all, decimal_place_value
        )
    except Exception as ex:
        raise RuntimeError('Failed to run diagnosis command.') from ex


def summary_command_handler(raw_data_file, rule_file, output_dir=None, output_file_format='md', decimal_place_value=2):
    """Run result summary.

    Args:
        raw_data_file (str): Path to raw data jsonl file.
        rule_file (str): Path to baseline yaml file.
        output_dir (str): Path to output directory.
        output_file_format (str): Format of the output file, 'excel', 'md' or 'html'. Defaults to 'md'.
        decimal_place_value (int): Number of decimal places to show in output.
    """
    try:
        # Create output directory
        sb_output_dir = create_sb_output_dir(output_dir)
        # Check arguments
        supported_output_format = ['excel', 'html', 'md']
        if output_file_format not in supported_output_format:
            raise CLIError('Output format must be in {}.'.format(str(supported_output_format)))
        check_argument_file('raw_data_file', raw_data_file)
        check_argument_file('rule_file', rule_file)
        # Run result summary
        ResultSummary().run(raw_data_file, rule_file, sb_output_dir, output_file_format, decimal_place_value)
    except Exception as ex:
        raise RuntimeError('Failed to run summary command.') from ex


def generate_baseline_command_handler(
    raw_data_file,
    summary_rule_file,
    diagnosis_rule_file=None,
    baseline_file=None,
    output_dir=None,
    decimal_place_value=2
):
    """Run result generate-baseline.

    If diagnosis_rule_file is None, use mean of the data as baseline.
    If diagnosis_rule_file is not None, use the rules in diagnosis_rule_file to execute fix_threshold algorithm.

    Args:
        raw_data_file (str): Path to raw data jsonl file.
        summary_rule_file (str): the file name of the summary rule file.
        diagnosis_rule_file (str): the file name of the diagnosis rules which used in fix_threshold algorithm.
        baseline_file (str): the file name of the previous baseline file that plan to merge with current baseline.
        output_dir (str): the directory to save the baseline file.
        decimal_place_value (int): the number of digits after the decimal point.
    """
    try:
        # Create output directory
        sb_output_dir = create_sb_output_dir(output_dir)
        # Check arguments
        check_argument_file('raw_data_file', raw_data_file)
        check_argument_file('rule_file', summary_rule_file)
        algorithm = 'mean'
        if diagnosis_rule_file:
            algorithm = 'fix_threshold'
            check_argument_file('rule_file', diagnosis_rule_file)
        if baseline_file:
            check_argument_file('baseline_file', baseline_file)
        # Run result generate-baseline
        BaselineGeneration().run(
            raw_data_file, summary_rule_file, diagnosis_rule_file, baseline_file, algorithm, sb_output_dir,
            decimal_place_value
        )
    except Exception as ex:
        raise RuntimeError('Failed to run generate-baseline command.') from ex


def sdc_check_command_handler(
    raw_data_file, output_dir=None, output_file_format='text', metric_pattern=None
):
    """Run SDC quorum check on multi-node per-step results.

    Compares per-step fingerprint values across nodes (same rank across nodes)
    using bit-exact equality, flagging any GPU that diverges as an SDC outlier.

    Args:
        raw_data_file (str): Path to results jsonl file containing multi-node per-step data.
        output_dir (str): Path to output directory.
        output_file_format (str): Format of output: 'text', 'json', or 'both'. Defaults to 'text'.
        metric_pattern (str): Metric substring for per-step data. Defaults to 'deterministic_loss_per_step'.
    """
    try:
        from superbench.analyzer.sdc_quorum import run_sdc_check

        fmt = output_file_format or 'text'
        supported_output_format = ['text', 'json', 'both']
        if fmt not in supported_output_format:
            raise CLIError('Output format must be in {}.'.format(str(supported_output_format)))
        check_argument_file('raw_data_file', raw_data_file)

        sb_output_dir = create_sb_output_dir(output_dir) if output_dir else None
        pattern = metric_pattern or 'deterministic_loss_per_step'

        result = run_sdc_check(raw_data_file, output_dir=sb_output_dir, output_format=fmt, metric_pattern=pattern)

        # Non-zero exit signals a positive SDC detection to callers / CI.
        if result.has_outliers:
            raise SystemExit(1)
    except (SystemExit, CLIError):
        raise
    except Exception as ex:
        raise RuntimeError('Failed to run sdc-check command.') from ex

