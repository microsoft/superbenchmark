# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network Utility."""

import socket
import re
import os
from pathlib import Path
from superbench.common.utils import logger


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
    if os.getenv('IB_DEVICES', None):
        ib_devices_env = os.getenv('IB_DEVICES').split(',')
        # Validate that IB_DEVICES contains either all
        # numeric indices or all device names, not mixed
        numeric_flags = [device.strip().isdigit() for device in ib_devices_env]
        all_numeric = all(numeric_flags)
        any_numeric = any(numeric_flags)

        # Check for mixed case (some numeric, some not)
        if any_numeric and not all_numeric:
            logger.log_and_raise(
                exception=ValueError,
                msg='IB_DEVICES contains mixed numeric indices and device names: {}. '
                'All values must be either numeric indices (e.g., "0,2,4,6") '
                'or device names (e.g., "mlx5_ib0,mlx5_ib2").'.format(os.getenv('IB_DEVICES'))
            )

        # If all numeric, fall through to discover actual devices; otherwise use provided names
        if not all_numeric:
            # All are device names, use them directly
            return ib_devices_env
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
