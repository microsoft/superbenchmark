# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The setuptools based setup module.

Reference:
    https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

import os
import sys
import pathlib
from typing import List, Tuple

from setuptools import setup, find_packages, Command

import superbench

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')


class Formatter(Command):
    """Cmdclass for `python setup.py format`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'format the code using yapf'
    user_options: List[Tuple[str, str, str]] = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Fromat the code using yapf."""
        errno = os.system('python3 -m yapf --in-place --recursive --exclude .git .')
        sys.exit(0 if errno == 0 else 1)


class Linter(Command):
    """Cmdclass for `python setup.py lint`.

    Args:
        Command (distutils.cmd.Command):
            Abstract base class for defining command classes.
    """

    description = 'lint the code using flake8'
    user_options: List[Tuple[str, str, str]] = []

    def initialize_options(self):
        """Set default values for options that this command supports."""
        pass

    def finalize_options(self):
        """Set final values for options that this command supports."""
        pass

    def run(self):
        """Lint the code with yapf, mypy, and flake8."""
        errno = os.system(
            ' && '.join(
                [
                    'python3 -m yapf --diff --recursive --exclude .git .',
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
    user_options: List[Tuple[str, str, str]] = []

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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: System :: Benchmark',
        'Topic :: System :: Clustering',
        'Topic :: System :: Hardware',
    ],
    keywords='benchmark, AI systems',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.6, <4',
    install_requires=[
        'ansible_base>=2.10.9;os_name=="posix"',
        'ansible_runner>=2.0.0rc1',
        'colorlog>=4.7.2',
        'jinja2>=2.10.1',
        'joblib>=1.0.1',
        'jsonlines>=2.0.0',
        'knack>=0.7.2',
        'matplotlib>=3.0.0',
        'natsort>=7.1.1',
        'openpyxl>=3.0.7',
        'omegaconf==2.0.6',
        'pandas>=1.1.5',
        'pyyaml>=5.3',
        'seaborn>=0.11.2',
        'tcping>=0.1.1rc1',
        'xlrd>=2.0.1',
        'xlsxwriter>=1.3.8',
        'xmltodict>=0.12.0',
    ],
    extras_require={
        'dev': ['pre-commit>=2.10.0'],
        'test': [
            'flake8-docstrings>=1.5.0',
            'flake8-quotes>=3.2.0',
            'flake8>=3.8.4',
            'mypy>=0.800',
            'pydocstyle>=5.1.1',
            'pytest-cov>=2.11.1',
            'pytest-subtests>=0.4.0',
            'pytest>=6.2.2',
            'types-pyyaml',
            'vcrpy>=4.1.1',
            'yapf==0.31.0',
        ],
        'nvidia': ['py3nvml>=0.2.6'],
        'ort': [
            'onnx>=1.10.2',
            'onnxruntime-gpu>=1.9.0',
        ],
        'torch': [
            'torch>=1.7.0a0',
            'torchvision>=0.8.0a0',
            'transformers>=4.3.3',
        ],
    },
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
