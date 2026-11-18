# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSeek-V2-Lite workers and helpers.

Workers run as ``python -m deepseek_v2_lite.<module>``, the inference ones in interpreters without SuperBench,
so every module except ``cache`` imports only the standard library and this package at load time.
"""
