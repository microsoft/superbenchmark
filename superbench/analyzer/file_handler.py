# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for file related functions in analyzer."""

from pathlib import Path
import re
import json

import jsonlines
import pandas as pd
import yaml

from superbench.common.utils import logger


def read_raw_data(raw_data_path):
    """Read raw data from raw_data_path and store them in raw_data_df.

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


def read_rules(rule_file=None):
    """Read rule from rule yaml file.

    Args:
        rule_file (str, optional): The path of rule yaml file. Defaults to None.

    Returns:
        dict: dict object read from yaml file
    """
    default_rule_file = Path(__file__).parent / 'rule/default_rule.yaml'
    p = Path(rule_file) if rule_file else default_rule_file
    if not p.is_file():
        logger.error('DataDiagnosis: invalid rule file path - {}'.format(str(p.resolve())))
        return None
    baseline = None
    with p.open() as f:
        baseline = yaml.load(f, Loader=yaml.SafeLoader)
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


def output_excel_raw_data(writer, raw_data_df, sheet_name):
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


def output_excel_data_not_accept(writer, data_not_accept_df, rules):
    """Output data_not_accept_df into 'Not Accept' excel page.

    Args:
        writer (xlsxwriter): xlsxwriter handle
        data_not_accept_df (DataFrame): the DataFrame to output
        rules (dict): the rules of DataDiagnosis
    """
    # Get the xlsxwriter workbook objects and init the format
    workbook = writer.book
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
                    col_index = columns.index(metric)
                    # Apply percent format for the columns whose rules are variance type.
                    if rules[rule]['function'] == 'variance':
                        worksheet.conditional_format(
                            row_start,
                            col_index,
                            row_end,
                            col_index,    # start_row, start_col, end_row, end_col
                            {
                                'type': 'no_blanks',
                                'format': percent_format
                            }
                        )
                    # Apply red format if the value violates the rule.
                    if rules[rule]['function'] == 'value' or rules[rule]['function'] == 'variance':
                        match = re.search(r'(>|<|<=|>=|==|!=)(.+)', rules[rule]['criteria'])
                        if not match:
                            continue
                        symbol = match.group(1)
                        condition = float(match.group(2))
                        worksheet.conditional_format(
                            row_start,
                            col_index,
                            row_end,
                            col_index,    # start_row, start_col, end_row, end_col
                            {
                                'type': 'cell',
                                'criteria': symbol,
                                'value': condition,
                                'format': color_format_red
                            }
                        )

        else:
            logger.warning('DataDiagnosis: excel_data_output - data_not_accept_df is empty.')
    else:
        logger.warning('DataDiagnosis: excel_data_output - data_not_accept_df is not DataFrame.')


def output_excel(raw_data_df, data_not_accept_df, output_path, rules):
    """Output the raw_data_df and data_not_accept_df results into excel file.

    Args:
        raw_data_df (DataFrame): raw data
        data_not_accept_df (DataFrame): defective nodes's detailed information
        output_path (str): the path of output excel file
        rules (dict): the rules of DataDiagnosis
    """
    try:
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        # Check whether writer is valiad
        if not isinstance(writer, pd.ExcelWriter):
            logger.error('DataDiagnosis: excel_data_output - invalid file path.')
            return
        output_excel_raw_data(writer, raw_data_df, 'Raw Data')
        output_excel_data_not_accept(writer, data_not_accept_df, rules)
        writer.save()
    except Exception as e:
        logger.error('DataDiagnosis: excel_data_output - {}'.format(str(e)))


def output_json_data_not_accept(data_not_accept_df, output_path):
    """Output data_not_accept_df into jsonl file.

    Args:
        data_not_accept_df (DataFrame): the DataFrame to output
        output_path (str): the path of output jsonl file
    """
    p = Path(output_path)
    try:
        data_not_accept_json = data_not_accept_df.to_json(orient='index')
        data_not_accept = json.loads(data_not_accept_json)
        if not isinstance(data_not_accept_df, pd.DataFrame):
            logger.warning('DataDiagnosis: output json data - data_not_accept_df is not DataFrame.')
            return
        if data_not_accept_df.empty:
            logger.warning('DataDiagnosis: output json data - data_not_accept_df is empty.')
            return
        with p.open('w') as f:
            for node in data_not_accept:
                line = data_not_accept[node]
                line['Index'] = node
                json_str = json.dumps(line)
                f.write(json_str + '\n')
    except Exception as e:
        logger.error('DataDiagnosis: output json data failed, msg: {}'.format(str(e)))
