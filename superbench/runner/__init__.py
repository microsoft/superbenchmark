# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench runner module."""

from superbench.common.utils.lazy_import import LazyImport

AnsibleClient = LazyImport('superbench.runner.ansible', 'AnsibleClient')
SuperBenchRunner = LazyImport('superbench.runner.runner', 'SuperBenchRunner')

__all__ = [
    'AnsibleClient',
    'SuperBenchRunner',
]
