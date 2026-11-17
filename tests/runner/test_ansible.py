# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Ansible Client test."""

import os
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from superbench.common.utils import LazyImport

AnsibleClient = LazyImport('superbench.runner.ansible', 'AnsibleClient')


class AnsiblePostinstallTestCase(unittest.TestCase):
    """Test the shared Makefile entry point without contacting Galaxy."""
    def test_bounded_collection_retries(self):
        """Stop on success and propagate exhausted installation failures."""
        for codes in [[0], [1, 0], [1, 1, 0], [17, 17, 17], [124, 124, 124]]:
            with self.subTest(codes=codes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                attempts = root / 'attempts'
                waits = root / 'waits'
                status_codes = root / 'codes'
                status_codes.write_text('\n'.join(map(str, codes)) + '\n')
                waits.touch()
                scripts = {
                    'ansible-galaxy': (
                        'printf "%s\\n" "$*" >> "$INSTALL_ATTEMPTS"\n'
                        'attempt=$(wc -l < "$INSTALL_ATTEMPTS")\n'
                        'exit "$(sed -n "${attempt}p" "$INSTALL_CODES")"\n'
                    ),
                    'sleep':
                    'printf "%s\\n" "$1" >> "$INSTALL_WAITS"\n',
                }
                for name, body in scripts.items():
                    script = root / name
                    script.write_text('#!/bin/sh\nset -eu\n' + body)
                    script.chmod(0o755)
                env = dict(
                    os.environ,
                    PATH=str(root) + os.pathsep + os.environ['PATH'],
                    INSTALL_ATTEMPTS=str(attempts),
                    INSTALL_WAITS=str(waits),
                    INSTALL_CODES=str(status_codes),
                )
                result = subprocess.run(
                    ['make', '--no-print-directory', 'postinstall'],
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode == 0, codes[-1] == 0, result.stdout)
                commands = attempts.read_text().splitlines()
                self.assertEqual(len(commands), len(codes), result.stdout)
                self.assertEqual(commands, [commands[0]] * len(codes))
                self.assertRegex(
                    commands[0],
                    r'^collection install ansible\.posix:\d+\.\d+\.\d+ '
                    r'ansible\.utils:\d+\.\d+\.\d+ community\.crypto:\d+\.\d+\.\d+$',
                )
                self.assertEqual(waits.read_text().splitlines(), ['10', '20'][:len(codes) - 1])
                if codes[-1]:
                    self.assertIn('attempt 3/3 failed (exit {}).'.format(codes[-1]), result.stdout)


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
        _, self.test_mpi_host_file = tempfile.mkstemp()

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        Path(self.host_file).unlink()
        Path(self.test_mpi_host_file).unlink()

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
                'host_pattern': '10.0.0.10',
            }
        )

    def test_update_mpi_config_for_different_inventory(self):
        """Test update_mpi_config of client for different inventory."""
        # Test for out-of-order
        with open(self.test_mpi_host_file, 'w') as fd:
            fd.write('all:\n  hosts:\n    10.0.0.12:\n    10.0.0.11:\n    10.0.0.10:\n    10.0.0.13:\n    10.0.0.14:\n')
        mess_hosts = AnsibleClient(
            OmegaConf.create(
                {
                    'host_file': self.test_mpi_host_file,
                    'host_username': 'user',
                    'host_password': 'pass',
                }
            )
        )
        self.assertDictEqual(
            mess_hosts.update_mpi_config(mess_hosts._config), {
                **mess_hosts._config,
                'host_pattern': '10.0.0.10',
            }
        )
        # Test for localhost
        with open(self.test_mpi_host_file, 'w') as fd:
            fd.write('all:\n  hosts:\n    localhost:\n')
        localhost = AnsibleClient(
            OmegaConf.create(
                {
                    'host_file': self.test_mpi_host_file,
                    'host_username': 'user',
                    'host_password': 'pass',
                }
            )
        )
        self.assertDictEqual(
            localhost.update_mpi_config(localhost._config), {
                **localhost._config,
                'host_pattern': 'localhost',
            }
        )
        # Test for no host
        with open(self.test_mpi_host_file, 'w') as fd:
            fd.write('all:\n  hosts:\n')
        no_hosts = AnsibleClient(
            OmegaConf.create(
                {
                    'host_file': self.test_mpi_host_file,
                    'host_username': 'user',
                    'host_password': 'pass',
                }
            )
        )
        self.assertDictEqual(
            no_hosts.update_mpi_config(no_hosts._config), {
                **no_hosts._config,
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

    def test_local_execution_and_failure_count(self):
        """Exercise the real runner and preserve nonzero command failures."""
        client = AnsibleClient(OmegaConf.create({'host_list': 'localhost'}))
        with tempfile.TemporaryDirectory(prefix='sb-ansible-client-') as directory:
            for command, expected in [('printf sb-ansible-smoke', 0), ('exit 7', 2)]:
                config = client.get_shell_config(command)
                config['extravars'] = {'ansible_python_interpreter': sys.executable}
                config['envvars'] = {
                    'ANSIBLE_LOCAL_TEMP': str(Path(directory) / 'local'),
                    'ANSIBLE_REMOTE_TEMP': str(Path(directory) / 'remote'),
                }
                self.assertEqual(client.run(config), expected)
        self.assertEqual(client.failure_count, 1)
