# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import pandas as pd

from superbench.common.utils import logger


def excel_raw_data_output(writer, raw_data_df):
    """Output raw data into 'Raw Data' excel page."""
    # Output the raw data
    if isinstance(raw_data_df, pd.DataFrame) and not raw_data_df.empty:
        raw_data_df.to_excel(writer, 'Raw Data', index=True)
    else:
        logger.warning('DataDiagnosis: excel_data_output - raw_data_df is empty.')


def excel_data_not_accept_output(writer, data_not_accept_df, baselilnes):
    """Output data_not_accept_df into 'Not Accept' excel page."""
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
                if baselilnes[colums]['rules']['name'] == 'variance':
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

                    if baselilnes[colums]['rules']['condition'] < 0:
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
                            'value': baselilnes[colums]['rules']['condition'],
                            'format': color_format_red
                        }
                    )    # Apply red format if the variance violates the rule.

                elif baselilnes[colums]['rules']['name'] == 'value':
                    worksheet.conditional_format(
                        row_start,
                        col_start,
                        row_end,
                        col_start,    # start_row, start_col, end_row, end_col
                        {
                            'type': 'cell',
                            'criteria': '>',
                            'value': baselilnes[colums]['criteria'],
                            'format': color_format_red
                        }
                    )    # Apply red format if the value violates the rule.

    else:
        logger.warning('DataDiagnosis: excel_data_output - data_not_accept_df is empty.')
