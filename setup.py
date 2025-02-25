# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os
import sys
import pathlib
from typing import List, Tuple, ClassVar

from setuptools import setup, find_packages, Command

import superbench

print(f'Python {sys.version_info.major}.{sys.version_info.minor} detected.')
if sys.version_info[:2] < (3, 11):
    import pkg_resources
    try:
        pkg_resources.require(['pip>=18', 'setuptools>=45, <66'])
    except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
        print(
            '\033[93mTry update pip/setuptools versions, for example, '
            'python3 -m pip install --upgrade pip wheel setuptools==65.7\033[0m'
        )
        raise

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')


class Formatter(Command):
    """Cmdclass for `python setup.py format`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'format the code using yapf'
    user_options: ClassVar[List[Tuple[str, str, str]]] = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Format the code using yapf."""
        if sys.version_info[:2] >= (3, 12):
            # TODO: Remove this block when yapf is compatible with Python 3.12+.
            print('Disable yapf for Python 3.12+ due to the compatibility issue.')
        else:
            errno = os.system('python3 -m yapf --in-place --recursive --exclude .git --exclude .eggs .')
            sys.exit(0 if errno == 0 else 1)


class Linter(Command):
    """Cmdclass for `python setup.py lint`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'lint the code using flake8'
    user_options: ClassVar[List[Tuple[str, str, str]]] = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Lint the code with yapf, mypy, and flake8."""
        if sys.version_info[:2] >= (3, 12):
            # TODO: Remove this block when yapf is compatible with Python 3.12+.
            print('Disable lint for Python 3.12+ due to the compatibility issue.')
        errno = os.system(
            ' && '.join(
                [
                    'python3 -m yapf --diff --recursive --exclude .git --exclude .eggs .' if sys.version_info[:2] <
                    (3, 12) else ':',
                    'python3 -m mypy .',
                    'python3 -m flake8',
                ]
            )
        )
        sys.exit(0 if errno == 0 else 1)


class Tester(Command):
    """Cmdclass for `python setup.py test`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'test the code using pytest'
    user_options: ClassVar[List[Tuple[str, str, str]]] = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Run pytest."""
        errno = os.system('python3 -m pytest -v --cov=superbench --cov-report=xml --cov-report=term-missing tests/')
        sys.exit(0 if errno == 0 else 1)


setup(
    name='superbench',
    version=superbench.__version__,
    description='Provide hardware and software benchmarks for AI systems.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/superbenchmark',
    author=superbench.__author__,
    author_email='superbench@microsoft.com',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: GPU',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: System :: Benchmark',
        'Topic :: System :: Clustering',
        'Topic :: System :: Hardware',
    ],
    keywords='benchmark, AI systems',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.7, <4',
    use_scm_version={
        'local_scheme': 'node-and-date',
        'version_scheme': lambda _: superbench.__version__,
        'fallback_version': f'{superbench.__version__}+unknown',
    },
    setup_requires=[
        'setuptools_scm',
    ],
    install_requires=[
        'ansible;os_name=="posix" and python_version>"3.10"',
        'ansible_base>=2.10.9;os_name=="posix" and python_version<="3.10"',
        'ansible_runner>=2.0.0rc1, <2.3.2;python_version<="3.10"',
        'ansible_runner;python_version>"3.10"',
        'colorlog>=6.7.0',
        'importlib_metadata',
        'jinja2>=2.10.1',
        'joblib>=1.0.1',
        'jsonlines>=2.0.0',
        'knack>=0.7.2',
        'markdown>=3.3.0',
        'matplotlib>=3.0.0',
        'natsort>=7.1.1',
        'networkx>=2.5',
        'numpy>=1.19.2',
        'omegaconf==2.3.0',
        'openpyxl>=3.0.7',
        'packaging>=21.0',
        'pandas>=1.1.5',
        'protobuf<=3.20.3',
        'pssh @ git+https://github.com/lilydjwg/pssh.git@v2.3.4',
        'pyyaml>=5.3',
        'requests>=2.27.1',
        'seaborn>=0.11.2',
        'tcping>=0.1.1rc1',
        'urllib3>=1.26.9',
        'xlrd>=2.0.1',
        'xlsxwriter>=1.3.8',
        'xmltodict>=0.12.0',
        'types-requests',
    ],
    extras_require=(
        lambda x: {
            **x,
            'develop': x['dev'] + x['test'],
            'cpuworker': x['torch'],
            'amdworker': x['torch'] + x['amd'],
            'nvworker': x['torch'] + x['ort'] + x['nvidia'],
        }
    )(
        {
            'dev': ['pre-commit>=2.10.0'],
            'test': [
                'flake8-docstrings>=1.5.0',
                'flake8-quotes>=3.2.0',
                'flake8>=3.8.4',
                'mypy>=0.800',
                'pydocstyle>=5.1.1',
                'pytest-cov>=2.11.1',
                'pytest-subtests>=0.4.0',
                'pytest>=6.2.2, <=7.4.4',
                'types-markdown',
                'types-setuptools',
                'types-pyyaml',
                'typing-extensions>=3.10',
                'urllib3<2.0',
                'vcrpy>=4.1.1',
                'yapf==0.31.0',
            ],
            'torch': [
                'safetensors==0.4.5',
                'tokenizers<=0.20.3',
                'torch>=1.7.0a0',
                'torchvision>=0.8.0a0',
                'transformers>=4.28.0',
            ],
            'ort': [
                'onnx>=1.10.2',
                'onnxruntime-gpu==1.12.0; python_version<"3.10" and platform_machine == "x86_64"',
                'onnxruntime-gpu; python_version>="3.10" and platform_machine == "x86_64"',
            ],
            'nvidia': ['py3nvml>=0.2.6'],
            'amd': ['amdsmi'],
        }
    ),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sb = superbench.cli.sb:main',
        ],
    },
    cmdclass={
        'format': Formatter,
        'lint': Linter,
        'test': Tester,
    },
    project_urls={
        'Source': 'https://github.com/microsoft/superbenchmark',
        'Tracker': 'https://github.com/microsoft/superbenchmark/issues',
    },
)
