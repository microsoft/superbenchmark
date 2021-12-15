# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Ansible Client test."""

import os
import unittest
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from superbench.runner.ansible import AnsibleClient


class AnsibleClientTestCase(unittest.TestCase):
    """A class for ansible client test cases."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        fd, self.host_file = tempfile.mkstemp()
        os.write(
            fd, (
                'all:\n'
                '  hosts:\n'
                '    10.0.0.10:\n'
                '    10.0.0.11:\n'
                '    10.0.0.12:\n'
                '    10.0.0.13:\n'
                '    10.0.0.14:\n'
            ).encode()
        )
        os.close(fd)

        self.ansible_client = AnsibleClient(
            OmegaConf.create({
                'host_file': self.host_file,
                'host_username': 'user',
                'host_password': 'pass',
            })
        )

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        Path(self.host_file).unlink()

    def test_init_config(self):
        """Test initial config of client."""
        self.assertDictEqual(
            self.ansible_client._config, {
                'host_pattern': 'all',
                'cmdline': f'--forks 5 --inventory {self.host_file} --user user --ask-pass --ask-become-pass',
                'passwords': {
                    'password': 'pass',
                    'passphrase': 'pass',
                },
            }
        )

    def test_update_mpi_config(self):
        """Test update_mpi_config of client."""
        self.assertDictEqual(
            self.ansible_client.update_mpi_config(self.ansible_client._config), {
                **self.ansible_client._config,
                'host_pattern': 'all[0]',
            }
        )

    def test_get_shell_config(self):
        """Test get_shell_config of client."""
        cmd = 'ls -la'
        self.assertDictEqual(
            self.ansible_client.get_shell_config(cmd), {
                'host_pattern': 'all',
                'cmdline': f'--forks 5 --inventory {self.host_file} --user user --ask-pass --ask-become-pass',
                'passwords': {
                    'password': 'pass',
                    'passphrase': 'pass',
                },
                'module': 'shell',
                'module_args': cmd,
            }
        )

    def test_get_playbook_config(self):
        """Test get_playbook_config of client."""
        self.assertDictEqual(
            self.ansible_client.get_playbook_config('play', {'foo': 'bar'}), {
                'host_pattern': 'all',
                'cmdline': f'--forks 5 --inventory {self.host_file} --user user --ask-pass --ask-become-pass',
                'passwords': {
                    'password': 'pass',
                    'passphrase': 'pass',
                },
                'extravars': {
                    'foo': 'bar',
                },
                'playbook': str(self.ansible_client._playbook_path / 'play'),
            }
        )
