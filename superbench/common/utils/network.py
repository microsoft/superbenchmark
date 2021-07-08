# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network Utility."""

import socket
import subprocess


def get_free_port():
    """Get a free port in current system.

    Return:
        port (int): a free port in current system.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    except OSError:
        return None
    finally:
        s.close()


def get_ib_devices():
    """Get available IB devices in the system and filter ethernet devices.

    Return:
        ib_devices (list): IB devices in current system.
    """
    command_get_devices = 'ls /sys/class/infiniband/'
    output = subprocess.run(
        command_get_devices,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        check=False,
        universal_newlines=True
    )
    devices = output.stdout.split()
    devices.sort()
    # Filter 'InfiniBand' devices by link_layer
    ib_devices = []
    for device in devices:
        command_get_ports = command_get_devices + device + '/ports/'
        output = subprocess.run(
            command_get_ports,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            check=False,
            universal_newlines=True
        )
        ports = output.stdout.split()
        for port in ports:
            command_get_link_layer = 'cat' + command_get_ports.split('ls')[1] + port + '/link_layer'
            output = subprocess.run(
                command_get_link_layer,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                check=False,
                universal_newlines=True
            )
            if 'InfiniBand' in output.stdout:
                ib_devices.append(device)
                break

    return ib_devices
