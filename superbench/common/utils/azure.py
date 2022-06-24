# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for Azure services."""

import urllib3


def get_azure_imds(path, format='text'):
    """Get metadata from Azure Instance Metadata Service.

    Args:
        path (str): URL path for Azure Instance Metadata Service.
        format (str, optional): Response format, text or json. Defaults to 'text'.

    Returns:
        str: Metadata in response. Defaults to '' if timeout or error occurs.
    """
    http = urllib3.PoolManager(
        headers={'Metadata': 'true'},
        timeout=urllib3.Timeout(connect=1.0, read=1.0),
        retries=urllib3.Retry(total=3, connect=0, backoff_factor=1.0),
    )
    try:
        r = http.request('GET', f'http://169.254.169.254/metadata/{path}?api-version=2020-06-01&format={format}')
        return r.data.decode('ascii')
    except Exception:
        return ''


def get_vm_size():
    """Get Azure VM SKU.

    Returns:
        str: Azure VM SKU.
    """
    return get_azure_imds('instance/compute/vmSize')
