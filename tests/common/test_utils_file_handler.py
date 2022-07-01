# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Azure services utilities."""

import unittest
from unittest import mock
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from superbench.common.utils import get_sb_config


class FileHandlerUtilsTestCase(unittest.TestCase):
    """A class for file_handler test cases."""
    @mock.patch('superbench.common.utils.azure.get_azure_imds')
    def test_get_sb_config_default(self, mock_get_azure_imds):
        """Test get_sb_config when no SKU detected, should use default config.

        Args:
            mock_get_azure_imds (function): Mock get_azure_imds function.
        """
        mock_get_azure_imds.return_value = ''
        with (Path.cwd() / 'superbench/config/default.yaml').open() as fp:
            self.assertEqual(get_sb_config(None), OmegaConf.create(yaml.load(fp, Loader=yaml.SafeLoader)))

    @mock.patch('superbench.common.utils.azure.get_azure_imds')
    def test_get_sb_config_sku(self, mock_get_azure_imds):
        """Test get_sb_config when SKU detected and config exists, should use corresponding config.

        Args:
            mock_get_azure_imds (function): Mock get_azure_imds function.
        """
        mock_get_azure_imds.return_value = 'Standard_NC96ads_A100_v4'
        with (Path.cwd() / 'superbench/config/azure/inference/standard_nc96ads_a100_v4.yaml').open() as fp:
            self.assertEqual(get_sb_config(None), OmegaConf.create(yaml.load(fp, Loader=yaml.SafeLoader)))

    @mock.patch('superbench.common.utils.azure.get_azure_imds')
    def test_get_sb_config_sku_nonexist(self, mock_get_azure_imds):
        """Test get_sb_config when SKU detected and no config exists, should use default config.

        Args:
            mock_get_azure_imds (function): Mock get_azure_imds function.
        """
        mock_get_azure_imds.return_value = 'Standard_Nonexist_A100_v4'
        with (Path.cwd() / 'superbench/config/default.yaml').open() as fp:
            self.assertEqual(get_sb_config(None), OmegaConf.create(yaml.load(fp, Loader=yaml.SafeLoader)))
