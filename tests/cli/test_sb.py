# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI command and scenario tests."""

import io
import contextlib
from functools import wraps
from knack.testsdk import ScenarioTest, StringCheck, NoneCheck

import superbench
from superbench.cli import SuperBenchCLI


def capture_system_exit(func):
    """Decorator to capture SystemExit in testing.

    Args:
        func (Callable): Decorated function.

    Returns:
        Callable: Decorator.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        f = io.StringIO()
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(f):
            func(self, *args, **kwargs)
        self.assertEqual(cm.exception.code, 2)
        self.stderr = f.getvalue()

    return wrapper


class SuperBenchCLIScenarioTest(ScenarioTest):
    """A class whose instances are CLI single test cases.

    Args:
        ScenarioTest (knack.testsdk.ScenarioTest): Test class for knack.
    """
    def __init__(self, method_name):
        """Override __init__ method for ScenarioTest.

        Args:
            method_name (str): ScenarioTest method_name.
        """
        sb_cli = SuperBenchCLI.get_cli()
        super().__init__(sb_cli, method_name)

    def test_sb_version(self):
        """Test sb version."""
        self.cmd('sb version', checks=[StringCheck(superbench.__version__)])

    def test_sb_deploy(self):
        """Test sb deploy."""
        self.cmd('sb deploy --host-list localhost', checks=[NoneCheck()])

    def test_sb_deploy_no_host(self):
        """Test sb deploy, no host_file or host_list provided, should fail."""
        self.cmd('sb deploy', expect_failure=True)

    def test_sb_exec(self):
        """Test sb exec."""
        self.cmd('sb exec --config-override superbench.enable=["none"]', checks=[NoneCheck()])

    def test_sb_run(self):
        """Test sb run."""
        self.cmd('sb run --host-list localhost --config-override superbench.enable=none', checks=[NoneCheck()])

    def test_sb_run_no_docker_auth(self):
        """Test sb run, only --docker-username argument, should fail."""
        result = self.cmd('sb run --docker-username test-user', expect_failure=True)
        self.assertEqual(result.exit_code, 1)

    def test_sb_run_no_host(self):
        """Test sb run, no --host-file or --host-list, should fail."""
        result = self.cmd('sb run --docker-image test:cuda11.1', expect_failure=True)
        self.assertEqual(result.exit_code, 1)

    def test_sb_run_nonexist_host_file(self):
        """Test sb run, --host-file does not exist, should fail."""
        result = self.cmd('sb run --host-file ./nonexist.yaml', expect_failure=True)
        self.assertEqual(result.exit_code, 1)

    def test_sb_node_info(self):
        """Test sb node info, should fail."""
        self.cmd('sb node info', expect_failure=False)
