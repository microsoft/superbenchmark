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
    description = 'format the code using yapf'
    user_options: List[Tuple[str, str, str]] = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = os.system('python3 -m yapf --in-place --recursive .')
        sys.exit(errno)


class Linter(Command):
    description = 'lint the code using flake8'
    user_options: List[Tuple[str, str, str]] = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = os.system(' && '.join([
            'python3 -m yapf --diff --recursive . ',
            'python3 -m mypy .',
            'python3 -m flake8',
        ]))
        sys.exit(errno)


class Tester(Command):
    description = 'test the code using pytest'
    user_options: List[Tuple[str, str, str]] = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = os.system('python3 -m pytest -v')
        sys.exit(errno)


setup(
    name='superbench',
    version=superbench.__version__,
    description='Provide hardware and software \
        benchmarks for AI systems and machines.',
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
        'dev': [],
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
