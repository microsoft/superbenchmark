# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Install or check the frozen Ansible environment without importing SuperBench."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / 'requirements'


def ansible_command(name):
    """Run an Ansible entry point with the same interpreter as the application."""
    script = Path(sysconfig.get_path('scripts')) / name
    if not script.is_file():
        found = shutil.which(name)
        if not found:
            raise RuntimeError('Cannot find {} for {}.'.format(name, sys.executable))
        script = Path(found)
    return [sys.executable, str(script)]


def install_collections(command):
    """Retry collection acquisition at most three times within a bounded budget."""
    for attempt in range(1, 4):
        try:
            result = subprocess.run(command, timeout=300)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            print('Collection installation exceeded 300 seconds.', file=sys.stderr, flush=True)
            returncode = 124
        if returncode == 0:
            return
        print(
            'Collection installation attempt {}/3 failed (exit {}).'.format(attempt, returncode),
            file=sys.stderr,
            flush=True,
        )
        if attempt == 3:
            raise subprocess.CalledProcessError(returncode, command)
        delay = attempt * 10
        print('Retrying in {} seconds.'.format(delay), file=sys.stderr, flush=True)
        time.sleep(delay)


def installed_packages():
    """Read installed distribution metadata without importing the engine."""
    packages = json.loads(
        subprocess.check_output(
            [sys.executable, '-m', 'pip', 'list', '--format=json', '--disable-pip-version-check'], text=True
        )
    )
    return {re.sub(r'[-_.]+', '-', package['name']).lower(): package['version'] for package in packages}


def check_python():
    """Fail if application installation changed any frozen Python dependency."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    installed = installed_packages()
    if 'ansible' in installed or ('ansible-base' in installed and 'ansible-core' in installed):
        raise RuntimeError('Use a clean environment: do not mix ansible, ansible-base and ansible-core distributions.')
    for line in (REQUIREMENTS / 'ansible.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        version = installed.get(canonicalize_name(requirement.name))
        if version is None or not requirement.specifier.contains(version):
            raise RuntimeError('{} is required, but {} is installed.'.format(requirement, version))
        print('{}=={}'.format(requirement.name, version))


def check_collections(requirements):
    """Check the versions, engine support and paths of the plugins actually loaded."""
    import yaml
    from ansible.release import __version__ as engine_version
    from packaging.specifiers import SpecifierSet

    expected = {item['name']: str(item['version']) for item in yaml.safe_load(requirements.read_text())['collections']}
    plugins = [
        ('module', 'community.crypto.openssh_keypair'),
        ('module', 'ansible.posix.synchronize'),
        ('lookup', 'ansible.utils.index_of'),
    ]
    for plugin_type, plugin in plugins:
        command = ansible_command('ansible-doc') + ['--json', '-t', plugin_type, plugin]
        documentation = json.loads(subprocess.check_output(command, text=True))
        if plugin not in documentation:
            raise RuntimeError('Cannot load required plugin {}.'.format(plugin))
        collection = Path(documentation[plugin]['doc']['filename']).resolve().parents[2]
        metadata = json.loads((collection / 'MANIFEST.json').read_text())['collection_info']
        name = '{}.{}'.format(metadata['namespace'], metadata['name'])
        if metadata['version'] != expected[name]:
            raise RuntimeError(
                '{}: expected {}, loaded {} from {}.'.format(name, expected[name], metadata['version'], collection)
            )
        runtime = yaml.safe_load((collection / 'meta' / 'runtime.yml').read_text())
        if not SpecifierSet(runtime['requires_ansible']).contains(engine_version):
            raise RuntimeError(
                '{} requires Ansible {}, not {}.'.format(name, runtime['requires_ansible'], engine_version)
            )
        for dependency, versions in metadata['dependencies'].items():
            if dependency not in expected or not SpecifierSet(versions).contains(expected[dependency]):
                raise RuntimeError('Pin the {} dependency of {} in {}.'.format(dependency, name, requirements))
        print('{}=={} ({})'.format(name, metadata['version'], collection))


def main():
    """Install the selected profile, or verify it without network access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check', action='store_true', help='Check installed versions and effective plugin paths only.'
    )
    args = parser.parse_args()
    profile = 'legacy' if sys.version_info[:2] < (3, 11) else 'modern'
    collections = REQUIREMENTS / 'ansible-{}.yml'.format(profile)
    try:
        if not args.check:
            incompatible = 'ansible-core' if profile == 'legacy' else 'ansible-base'
            if {'ansible', incompatible}.intersection(installed_packages()):
                raise RuntimeError('Create a clean environment before changing Ansible distributions.')
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r',
                 str(REQUIREMENTS / 'ansible.txt')], check=True
            )
        check_python()
        if not args.check:
            install_collections(
                ansible_command('ansible-galaxy') +
                ['collection', 'install', '--server', 'https://galaxy.ansible.com', '-r',
                 str(collections)]
            )
        check_collections(collections)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print('Ansible dependency setup failed: {}'.format(error), file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
