# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unittest TestCase helpers."""

import os
import shutil
import tempfile
from pathlib import Path


class BenchmarkTestCase(object):
    """Base class for benchmark test case.

    Examples:
        Inherit from both BenchmarkTestCase and unittest.TestCase.
        ```
        def FooBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
            def setUp(self):
                super().setUp()
                ...
        ```
    """
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        pass

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class.

        Will create a temp directory and mock envs for all tests.
        Run once for the whole class.
        """
        cls._tmp_dir = tempfile.mkdtemp(prefix='sbtest')
        cls._curr_mock_envs = {}

    @classmethod
    def tearDownClass(cls):
        """Hook method for deconstructing the class fixture after running all tests in the class.

        Will restore original envs and cleanup temp directory.
        Run once for the whole class.
        """
        cls.cleanupMockEnvs(cls)
        shutil.rmtree(cls._tmp_dir)

    def createMockEnvs(self, envs=None):
        """Create mock envs for tests.

        Args:
            envs (dict, optional): Environment variables to be mocked.
                Defaults to None and will mock SB_MICRO_PATH to temp directory.
        """
        if not envs:
            envs = {'SB_MICRO_PATH': self._tmp_dir}
        for name in envs:
            self._curr_mock_envs[name] = os.environ.get(name, None)
            os.environ[name] = envs[name]

    def cleanupMockEnvs(self):
        """Cleanup mock envs and restore original envs."""
        for name in self._curr_mock_envs:
            if self._curr_mock_envs[name] is None:
                del os.environ[name]
            else:
                os.environ[name] = self._curr_mock_envs[name]

    def createMockFiles(self, files, mode=0o755):
        """Create mock files for tests.

        Args:
            files (List[str]): List of file names, relative path will be created under temp directory.
            mode (int, optional): Octal integer for file mode. Defaults to 0o755.
        """
        for filename in files:
            filepath = Path(self._tmp_dir) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.touch(mode=mode, exist_ok=True)
