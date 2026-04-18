# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Metric sort helpers for analyzer outputs.

This module keeps benchmark-specific metric ordering isolated from the generic
summary generation flow. Benchmarks without a registered sorter fall back to
plain string ordering.
"""

import re

_RCCL_PATTERN = re.compile(r'^(?P<bench>rccl-bw(?::[^/]+)?)/(?P<op>[^_]+)_(?P<size>\d+)_(?P<suffix>.+?)(?::\d+)?$')
_HPCG_PATTERN = re.compile(r'^(?P<bench>gpu-hpcg(?::[^/]+)?)/(?P<metric>.+?)(?::\d+)?$')

_HPCG_METRIC_ORDER = {
    'local_domain_x': 0,
    'local_domain_y': 1,
    'local_domain_z': 2,
    'global_domain_x': 3,
    'global_domain_y': 4,
    'global_domain_z': 5,
    'process_domain_x': 6,
    'process_domain_y': 7,
    'process_domain_z': 8,
    'total_time': 9,
    'setup_time': 10,
    'optimization_time': 11,
    'ddot_gflops': 12,
    'ddot_bandwidth': 13,
    'ddot_gflops_per_process': 14,
    'ddot_bandwidth_per_process': 15,
    'waxpby_gflops': 16,
    'waxpby_bandwidth': 17,
    'waxpby_gflops_per_process': 18,
    'waxpby_bandwidth_per_process': 19,
    'spmv_gflops': 20,
    'spmv_bandwidth': 21,
    'spmv_gflops_per_process': 22,
    'spmv_bandwidth_per_process': 23,
    'mg_gflops': 24,
    'mg_bandwidth': 25,
    'mg_gflops_per_process': 26,
    'mg_bandwidth_per_process': 27,
    'total_gflops': 28,
    'total_bandwidth': 29,
    'total_gflops_per_process': 30,
    'total_bandwidth_per_process': 31,
    'final_gflops': 32,
    'final_bandwidth': 33,
    'final_gflops_per_process': 34,
    'final_bandwidth_per_process': 35,
    'is_valid': 36,
}


def _rccl_sort_key(metric_name):
    """Sort RCCL metrics by benchmark, operation, then numeric message size."""
    match = _RCCL_PATTERN.match(metric_name)
    if not match:
        return None

    return (
        0,
        match.group('bench'),
        match.group('op'),
        int(match.group('size')),
        match.group('suffix'),
        metric_name,
    )


def _hpcg_sort_key(metric_name):
    """Sort HPCG metrics roughly in the order they appear in rocHPCG logs."""
    match = _HPCG_PATTERN.match(metric_name)
    if not match:
        return None

    metric = match.group('metric')
    return (
        1,
        match.group('bench'),
        _HPCG_METRIC_ORDER.get(metric, 999),
        metric,
        metric_name,
    )


_SORTERS = (
    _rccl_sort_key,
    _hpcg_sort_key,
)


def sort_metrics(metrics):
    """Sort metrics with benchmark-specific sorters and a stable default fallback."""
    def sort_key(metric_name):
        for sorter in _SORTERS:
            key = sorter(metric_name)
            if key is not None:
                return key
        return (999, metric_name)

    return sorted(metrics, key=sort_key)
