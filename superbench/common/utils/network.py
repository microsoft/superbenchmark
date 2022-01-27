# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network Utility."""

import socket
import re
from pathlib import Path


def get_free_port():
    """Get a free port in current system.

    Return:
        port (int): a free port in current system.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    except OSError:
        return None
    finally:
        s.close()


def get_ib_devices():
    """Get available IB devices with available ports in the system and filter ethernet devices.

    Return:
        ib_devices_port (list): IB devices with available ports in current system.
    """
    devices = list(p.name for p in Path('/sys/class/infiniband').glob('*'))
    ib_devices_port_dict = {}
    for device in devices:
        ports = list(p.name for p in (Path('/sys/class/infiniband') / device / 'ports').glob('*'))
        ports.sort(key=lambda s: [int(ch) if ch.isdigit() else ch for ch in re.split(r'(\d+)', s)])
        for port in ports:
            with (Path('/sys/class/infiniband') / device / 'ports' / port / 'link_layer').open('r') as f:
                # Filter 'InfiniBand' devices by link_layer
                if f.read().strip() == 'InfiniBand':
                    if device not in ib_devices_port_dict:
                        ib_devices_port_dict[device] = [port]
                    else:
                        ib_devices_port_dict[device].append(port)
    ib_devices = list(ib_devices_port_dict.keys())
    ib_devices.sort(key=lambda s: [int(ch) if ch.isdigit() else ch for ch in re.split(r'(\d+)', s)])
    ib_devices_port = []
    for device in ib_devices:
        ib_devices_port.append(device + ':' + ','.join(ib_devices_port_dict[device]))
    return ib_devices_port
