# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for dependency acquisition and the installed Ansible collections."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    'install_ansible',
    Path(__file__).resolve().parents[2] / 'scripts' / 'install_ansible.py'
)
assert spec is not None and spec.loader is not None
install_ansible = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install_ansible)


class CollectionRetryTest(unittest.TestCase):
    """Exercise bounded retries without contacting Galaxy or waiting."""
    def test_success_and_recovery(self):
        """Stop on the first successful attempt and keep the original command."""
        command = ['ansible-galaxy', 'collection', 'install', '-r', 'requirements.yml']
        for codes, delays in [([0], []), ([1, 0], [10]), ([1, 1, 0], [10, 20])]:
            with self.subTest(codes=codes), mock.patch.object(
                install_ansible.subprocess,
                'run',
                side_effect=[subprocess.CompletedProcess(command, code) for code in codes]
            ) as run, mock.patch.object(install_ansible.time, 'sleep') as sleep:
                install_ansible.install_collections(command)
                self.assertEqual(run.call_args_list, [mock.call(command, timeout=300)] * len(codes))
                self.assertEqual(sleep.call_args_list, [mock.call(delay) for delay in delays])

    def test_exhaustion_preserves_failure(self):
        """Propagate the last failure rather than reporting a successful install."""
        command = ['ansible-galaxy']
        with mock.patch.object(
            install_ansible.subprocess, 'run', return_value=subprocess.CompletedProcess(command, 17)
        ) as run, mock.patch.object(install_ansible.time, 'sleep') as sleep:
            with self.assertRaises(subprocess.CalledProcessError) as error:
                install_ansible.install_collections(command)
            self.assertEqual(error.exception.returncode, 17)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sleep.call_args_list, [mock.call(10), mock.call(20)])

    def test_timeout_is_bounded(self):
        """Terminate timed-out acquisition attempts and fail after the budget."""
        command = ['ansible-galaxy']
        with mock.patch.object(install_ansible.subprocess, 'run', side_effect=subprocess.TimeoutExpired(command, 300)
                               ) as run, mock.patch.object(install_ansible.time, 'sleep'):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                install_ansible.install_collections(command)
            self.assertEqual(error.exception.returncode, 124)
            self.assertEqual(run.call_count, 3)

    def test_changed_python_dependency_is_rejected(self):
        """Detect a dependency changed by a later application installation."""
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory)
            (requirements / 'ansible.txt').write_text('packaging==26.3\n')
            with mock.patch.object(install_ansible, 'REQUIREMENTS', requirements), mock.patch.object(
                install_ansible.subprocess, 'check_output', return_value='[{"name":"packaging","version":"24.0"}]'
            ):
                with self.assertRaisesRegex(RuntimeError, 'packaging==26.3'):
                    install_ansible.check_python()

    def test_conflicting_distribution_is_rejected_before_install(self):
        """Do not overwrite another distribution sharing the ansible namespace."""
        with mock.patch.object(sys, 'argv', [
            'install_ansible.py'
        ]), mock.patch.object(install_ansible, 'installed_packages',
                              return_value={'ansible':
                                            '1.0'}), mock.patch.object(install_ansible.subprocess, 'run') as run:
            self.assertEqual(install_ansible.main(), 1)
            run.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'Ansible collections require POSIX.')
    def test_shadowed_collection_is_rejected(self):
        """Reject a different collection loaded ahead of the pinned version."""
        with tempfile.TemporaryDirectory() as directory:
            collection = Path(directory)
            (collection / 'MANIFEST.json').write_text(
                json.dumps(
                    {
                        'collection_info': {
                            'namespace': 'community',
                            'name': 'crypto',
                            'version': '0.0.0',
                            'dependencies': {}
                        }
                    }
                )
            )
            documentation = {
                'community.crypto.openssh_keypair': {
                    'doc': {
                        'filename': str(collection / 'plugins' / 'modules' / 'openssh_keypair.py')
                    }
                }
            }
            with mock.patch.object(install_ansible.subprocess, 'check_output', return_value=json.dumps(documentation)):
                with self.assertRaisesRegex(RuntimeError, 'loaded 0.0.0'):
                    install_ansible.check_collections(install_ansible.REQUIREMENTS / 'ansible-modern.yml')


@unittest.skipUnless(os.name == 'posix', 'Ansible collections require POSIX.')
class InstalledCollectionsTest(unittest.TestCase):
    """Run the real plugins on disposable local files, without Docker or SSH."""
    def test_installed_collections(self):
        """Check key generation, rank lookup and the actual result-fetch playbook."""
        with tempfile.TemporaryDirectory(prefix='sb-ansible-test-') as directory:
            config = Path(directory) / 'ansible.cfg'
            config.write_text('[defaults]\n')
            command = install_ansible.ansible_command('ansible-playbook') + [
                '-i',
                'localhost,',
                '-c',
                'local',
                str(Path(__file__).with_name('test_dependencies.yaml')),
                '-e',
                json.dumps({
                    'test_root': directory,
                    'ansible_python_interpreter': sys.executable
                }),
            ]
            env = dict(os.environ, ANSIBLE_CONFIG=str(config), ANSIBLE_LOCAL_TEMP=str(Path(directory) / 'local-tmp'))
            result = subprocess.run(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120
            )
            self.assertEqual(result.returncode, 0, result.stdout)
