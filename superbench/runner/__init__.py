# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench runner module."""

from superbench.common.utils.lazy_import import LazyImport
from superbench.runner.runner import SuperBenchRunner

AnsibleClient = LazyImport('superbench.runner.ansible', 'AnsibleClient')

__all__ = [
    'AnsibleClient',
    'SuperBenchRunner',
]
