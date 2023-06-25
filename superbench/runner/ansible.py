# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Ansible Client."""

import tempfile
from pathlib import Path

import ansible_runner
from ansible.parsing.dataloader import DataLoader
from ansible.inventory.manager import InventoryManager

from superbench.common.utils import logger


class AnsibleClient():
    """Ansible Client class."""
    def __init__(self, config):
        """Initilize.

        Args:
            config (DictConfig): Ansible config object.
        """
        self._playbook_path = Path(__file__).parent / 'playbooks'
        self._config = {
            'host_pattern': 'localhost',
            'cmdline': '--forks 128',
        }
        self._head_host = None
        self.failure_count = 0
        if config:
            inventory_file = getattr(config, 'host_file', None)
            inventory_list = getattr(config, 'host_list', None)
            if inventory_list:
                inventory_list = inventory_list.strip(',')
            if inventory_file or inventory_list:
                self._config['host_pattern'] = 'all'
                inventory = InventoryManager(loader=DataLoader(), sources=inventory_file or f'{inventory_list},')
                host_list = inventory.get_hosts(pattern='all', order='sorted')
                if len(host_list) > 0:
                    self._config['cmdline'] = '--forks {}'.format(len(host_list))
                    self._head_host = host_list[0].get_name()
                if inventory_list in ['localhost', '127.0.0.1']:
                    self._config['cmdline'] += ' --connection local'
                self._config['cmdline'] += ' --inventory {}'.format(inventory_file or f'{inventory_list},')
            username = getattr(config, 'host_username', None)
            if username:
                self._config['cmdline'] += ' --user {}'.format(username)
            password = getattr(config, 'host_password', None)
            if password:
                self._config['passwords'] = {
                    'password': password,
                    'passphrase': password,
                }
            key_file = getattr(config, 'private_key', None)
            if key_file:
                self._config['cmdline'] += ' --private-key {}'.format(key_file)
            elif password:
                self._config['cmdline'] += ' --ask-pass --ask-become-pass'
        logger.info(self._config)

    def run(self, ansible_config, cancel_callback=None, sudo=False):    # pragma: no cover
        """Run Ansible runner.

        Args:
            ansible_config (dict): Ansible config dict.
            cancel_callback (Callable): Ansible runner cancel callback.
            sudo (bool): Run as sudo or not. Defaults to False.

        Returns:
            int: Ansible return code.
        """
        if sudo:
            logger.info('Run as sudo ...')
            ansible_config['cmdline'] += ' --become'
        with tempfile.TemporaryDirectory(prefix='ansible') as tmpdir:
            r = ansible_runner.run(private_data_dir=tmpdir, cancel_callback=cancel_callback, **ansible_config)
            logger.debug(r.stats)
        if r.rc == 0:
            logger.info('Run succeed, return code {}.'.format(r.rc))
        else:
            self.failure_count += 1
            logger.warning('Run failed, return code {}.'.format(r.rc))
        return r.rc

    def update_mpi_config(self, ansible_config):
        """Update ansible config for mpi, run on the first host of inventory group.

        Args:
            ansible_config (dict): Ansible config dict.

        Returns:
            dict: Updated Ansible config dict.
        """
        if not self._head_host:
            ansible_config['host_pattern'] += '[0]'
        else:
            ansible_config['host_pattern'] = self._head_host
        return ansible_config

    def get_shell_config(self, cmd):
        """Get ansible config for shell module.

        Args:
            cmd (str): Shell command for config.

        Returns:
            dict: Ansible config dict.
        """
        logger.info('Run {} on remote ...'.format(cmd))
        ansible_config = {
            **self._config,
            'module': 'shell',
            'module_args': cmd,
        }
        return ansible_config

    def get_playbook_config(self, playbook, extravars=None):
        """Get ansible config for playbook.

        Args:
            playbook (str): Playbook file name.
            extravars (dict): Extra variables in playbook. Defaults to None.

        Returns:
            dict: Ansible config dict.
        """
        logger.info('Run playbook {} ...'.format(playbook))
        ansible_config = {
            **self._config,
            'extravars': extravars,
            'playbook': str(self._playbook_path / playbook),
        }
        return ansible_config
