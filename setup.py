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
        sys.exit(errno)


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
        sys.exit(errno)


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
        errno = os.system('python3 -m pytest -v')
        sys.exit(errno)


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
    install_requires=[],
    extras_require={
        'dev': ['pre-commit'],
        'test': [
            'yapf',
            'mypy',
            'flake8',
            'flake8-quotes',
            'flake8-docstrings',
            'pytest',
        ],
    },
    package_data={},
    entry_points={
        'console_scripts': [],
    },
    cmdclass={
        'format': Formatter,
        'lint': Linter,
        'test': Tester,
    },
)
