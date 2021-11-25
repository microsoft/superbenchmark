# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import pandas as pd

from superbench.common.utils import logger


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


def excel_data_not_accept_output(writer, data_not_accept_df, baselines):
    """Output data_not_accept_df into 'Not Accept' excel page.

    Args:
        writer (xlsxwriter): xlsxwriter handle
        data_not_accept_df (DataFrame): the DataFrame to output
        baselines (dict): the baseline to diagnosis data
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
            fix_table_len = 4
            columns = data_not_accept_df.columns
            columns = columns[fix_table_len:]
            col_start = fix_table_len
            # Get the xlsxwriter worksheet objects.
            worksheet = writer.sheets['Not Accept']

            for colums in columns:
                col_start += 1
                if baselines[colums]['rules']['name'] == 'variance':
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

                    if baselines[colums]['rules']['condition'] < 0:
                        symbol = '<='
                    else:
                        symbol = '>='
                    worksheet.conditional_format(
                        row_start,
                        col_start,
                        row_end,
                        col_start,    # start_row, start_col, end_row, end_col
                        {
                            'type': 'cell',
                            'criteria': symbol,
                            'value': baselines[colums]['rules']['condition'],
                            'format': color_format_red
                        }
                    )    # Apply red format if the variance violates the rule.

                elif baselines[colums]['rules']['name'] == 'value':
                    worksheet.conditional_format(
                        row_start,
                        col_start,
                        row_end,
                        col_start,    # start_row, start_col, end_row, end_col
                        {
                            'type': 'cell',
                            'criteria': '>',
                            'value': baselines[colums]['criteria'],
                            'format': color_format_red
                        }
                    )    # Apply red format if the value violates the rule.

    else:
        logger.warning('DataDiagnosis: excel_data_output - data_not_accept_df is empty.')
