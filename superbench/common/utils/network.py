# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network Utility."""

import socket
import subprocess
from contextlib import closing


def get_free_port():
    """Get a free port in current system.

    Return:
        port (int): a free port in current system.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_ib_devices():
    """Get available IB devices in the system and filter ethernet devices.

    Return:
        ib_devices (list): IB devices in current system.
    """
    # Filter 'InfiniBand' devices by link_layer
    command = "ibv_devinfo | awk '$1 ~ /hca_id/||/link_layer:/ {print $1,$2}'"
    output = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=False, universal_newlines=True
    )
    lines = list(filter(None, output.stdout.replace('\n', ' ').split('hca_id:')))
    ib_devices = []
    for line in lines:
        if 'InfiniBand' in line:
            ib_devices.append(line.split()[0])
    ib_devices.sort()

    return ib_devices
