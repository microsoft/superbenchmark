# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build numactl command fragments for runner modes."""

from omegaconf import ListConfig

GPU_AFFINITY = 'gpu_affinity'
GPU_NUMA_AFFINITY_ENV = 'SB_GPU_NUMA_AFFINITY'


def _format_template_value(value, mode):
    """Format a mode template value."""
    if isinstance(value, str):
        return value.format(proc_rank=mode.proc_rank, proc_num=mode.proc_num)
    if isinstance(value, (list, tuple, ListConfig)):
        return ','.join(_format_template_value(item, mode) for item in value)
    return str(value)


def _is_disabled_value(value):
    """Return whether a config value disables the corresponding option."""
    return value is None or value is False or (isinstance(value, str) and value.lower() in ['none', 'null', 'false'])


def _resolve_node_value(value, mode):
    """Resolve a numactl NUMA-node value.

    Args:
        value: numactl node binding config value.
        mode (DictConfig): Runner mode.

    Returns:
        tuple[str | None, bool]: Resolved value and whether it uses GPU affinity.
    """
    if _is_disabled_value(value):
        return None, False
    if isinstance(value, str) and value.lower() == GPU_AFFINITY:
        return '${%s}' % GPU_NUMA_AFFINITY_ENV, True
    return _format_template_value(value, mode), False


def _resolve_cpu_value(value, mode):
    """Resolve a numactl CPU-list value."""
    if _is_disabled_value(value):
        return None
    if isinstance(value, str) and value.lower() == GPU_AFFINITY:
        raise ValueError('gpu_affinity is not supported for numactl.physcpubind.')
    return _format_template_value(value, mode)


def get_local_numactl_command(mode):
    """Get setup and numactl command fragments for local mode.

    Args:
        mode (DictConfig): Runner mode.

    Returns:
        tuple[str, str]: Setup command and numactl command.
    """
    if 'numactl' not in mode:
        return '', ''

    numactl_config = mode.numactl
    if numactl_config is None:
        return '', ''

    cpunodebind, cpunodebind_uses_gpu = _resolve_node_value(numactl_config.get('cpunodebind', None), mode)
    membind, membind_uses_gpu = _resolve_node_value(numactl_config.get('membind', None), mode)
    physcpubind = _resolve_cpu_value(numactl_config.get('physcpubind', None), mode)
    if cpunodebind is None and membind is None and physcpubind is None:
        return '', ''

    setup_command = ''
    if cpunodebind_uses_gpu or membind_uses_gpu:
        gpu_id = _format_template_value(numactl_config.get('gpu_id', '{proc_rank}'), mode)
        setup_command = '{}=$(sb node topo --get gpu-numa-affinity --gpu-id {})'.format(
            GPU_NUMA_AFFINITY_ENV,
            gpu_id,
        )

    numactl_parts = ['numactl']
    if cpunodebind is not None:
        numactl_parts.extend(['-N', cpunodebind])
    if membind is not None:
        numactl_parts.extend(['-m', membind])
    if physcpubind is not None:
        numactl_parts.extend(['-C', physcpubind])

    return setup_command, ' '.join(numactl_parts)
