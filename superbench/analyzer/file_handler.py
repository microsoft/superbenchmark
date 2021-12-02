# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

from pathlib import Path

import json
import jsonlines
import pandas as pd
import yaml

from superbench.common.utils import logger


def read_raw_data(raw_data_path):
    """Read raw data from raw_data_path and store them in self._raw_data_df.

    Args:
        raw_data_path (str): the path of raw data jsonl file
    Returns:
        DataFrame: raw data, node as index, metric name as columns
    """
    p = Path(raw_data_path)
    raw_data_df = pd.DataFrame()
    if not p.is_file():
        logger.error('DataDiagnosis: invalid raw data path - {}'.format(raw_data_path))
        return raw_data_df

    try:
        with p.open(encoding='utf-8') as f:
            for single_node_summary in jsonlines.Reader(f):
                raw_data_df = raw_data_df.append(single_node_summary, ignore_index=True)
        raw_data_df = raw_data_df.rename(raw_data_df['node'])
        raw_data_df = raw_data_df.drop(columns=['node'])
    except Exception as e:
        logger.error('Analyzer: invalid raw data fomat - {}'.format(str(e)))
    return raw_data_df


def read_rules(baseline_file=None):
    """Read rule from rule yaml file.

    Args:
        baseline_file (str, optional): The path of rule yaml file. Defaults to None.

    Returns:
        dict: dict object read from yaml file
    """
    default_rule_file = Path(__file__).parent / 'default_rule.yaml'
    p = Path(baseline_file) if baseline_file else default_rule_file
    if not p.is_file():
        logger.error('DataDiagnosis: invalid rule file path - {}'.format(str(p.resolve())))
        return None
    baseline = None
    with p.open() as f:
        baseline = yaml.load(f)
    return baseline


def read_baseline(baseline_file):
    """Read baseline from baseline json file.

    Args:
        baseline_file (str): The path of baseline json file.

    Returns:
        dict: dict object read from json file
    """
    p = Path(baseline_file)
    if not p.is_file():
        logger.error('DataDiagnosis: invalid baseline file path - {}'.format(str(p.resolve())))
        return None
    baseline = None
    with p.open() as f:
        baseline = json.load(f)
    return baseline


def excel_raw_data_output(writer, raw_data_df, sheet_name):
    """Output raw data into 'sheet_name' excel page.

    Args:
        writer (xlsxwriter): xlsxwriter handle
        raw_data_df (DataFrame): the DataFrame to output
        sheet_name (str): sheet name of the excel
    """
    # Output the raw data
    if isinstance(raw_data_df, pd.DataFrame) and not raw_data_df.empty:
        raw_data_df.to_excel(writer, sheet_name, index=True)
    else:
        logger.warning('DataDiagnosis: excel_data_output - {} data_df is empty.'.format(sheet_name))


def excel_data_not_accept_output(writer, data_not_accept_df, rules):
    """Output data_not_accept_df into 'Not Accept' excel page.

    Args:
        writer (xlsxwriter): xlsxwriter handle
        data_not_accept_df (DataFrame): the DataFrame to output
        rules (dict): the rules to diagnosis data
    """
    # Get the xlsxwriter workbook objects and init the color format
    workbook = writer.book
    # Add a format. red fill with dark red text.
    color_format_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
    percent_format = workbook.add_format({'num_format': '0.00%'})

    # Output the not accept
    if isinstance(data_not_accept_df, pd.DataFrame):
        data_not_accept_df.to_excel(writer, 'Not Accept', index=True)
        if not data_not_accept_df.empty:
            row_start = 1
            row_end = max(row_start, len(data_not_accept_df))
            columns = list(data_not_accept_df.columns)
            worksheet = writer.sheets['Not Accept']

            for rule in rules:
                for metric in rules[rule]['metrics']:
                    col_start = columns.index(metric)
                    symbol = rules[rule]['criteria'].split(',')[0]
                    if rules[rule]['function'] == 'variance':
                        worksheet.conditional_format(
                            row_start,
                            col_start,
                            row_end,
                            col_start,    # start_row, start_col, end_row, end_col
                            {
                                'type': 'no_blanks',
                                'format': percent_format
                            }
                        )    # Apply percent format for the columns whose rules are variance type.
                        condition = rules[rule]['criteria'].split(',')[1]
                        if '%' in condition:
                            condition = float(condition.strip('%')) / 100
                        else:
                            condition = float(condition)
                        worksheet.conditional_format(
                            row_start,
                            col_start,
                            row_end,
                            col_start,    # start_row, start_col, end_row, end_col
                            {
                                'type': 'cell',
                                'criteria': symbol,
                                'value': condition,
                                'format': color_format_red
                            }
                        )    # Apply red format if the variance violates the rule.

                    elif rules[rule]['function'] == 'value':
                        condition = float(rules[rule]['criteria'].split(',')[1])
                        worksheet.conditional_format(
                            row_start,
                            col_start,
                            row_end,
                            col_start,    # start_row, start_col, end_row, end_col
                            {
                                'type': 'cell',
                                'criteria': symbol,
                                'value': condition,
                                'format': color_format_red
                            }
                        )    # Apply red format if the value violates the rule.

    else:
        logger.warning('DataDiagnosis: excel_data_output - data_not_accept_df is empty.')
