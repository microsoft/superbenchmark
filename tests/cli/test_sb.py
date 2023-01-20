# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI command and scenario tests."""

import io
import contextlib
from functools import wraps
from knack.testsdk import ScenarioTest, StringContainCheck, NoneCheck, JMESPathCheck
from pathlib import Path
from unittest import mock

import superbench
from superbench.cli import SuperBenchCLI
from superbench.benchmarks import BenchmarkRegistry


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
        self.cmd('sb version', checks=[StringContainCheck(superbench.__version__)])

    @mock.patch('superbench.runner.SuperBenchRunner.get_failure_count')
    def test_sb_deploy(self, mocked_failure_count):
        """Test sb deploy."""
        mocked_failure_count.return_value = 0
        self.cmd('sb deploy --host-list localhost', checks=[NoneCheck()])

    @mock.patch('superbench.runner.SuperBenchRunner.get_failure_count')
    def test_sb_deploy_skippull(self, mocked_failure_count):
        """Test sb deploy without docker pull."""
        mocked_failure_count.return_value = 0
        self.cmd('sb deploy --host-list localhost --no-image-pull', checks=[NoneCheck()])

    def test_sb_deploy_no_host(self):
        """Test sb deploy, no host_file or host_list provided, should fail."""
        self.cmd('sb deploy', expect_failure=True)

    def test_sb_exec(self):
        """Test sb exec."""
        self.cmd('sb exec --config-override superbench.enable=["none"]', checks=[NoneCheck()])

    @mock.patch('superbench.runner.SuperBenchRunner.get_failure_count')
    def test_sb_run(self, mocked_failure_count):
        """Test sb run."""
        mocked_failure_count.return_value = 0
        self.cmd('sb run --host-list localhost --config-override superbench.enable=none', checks=[NoneCheck()])

    @mock.patch('superbench.runner.SuperBenchRunner.get_failure_count')
    def test_sb_run_skipdocker(self, mocked_failure_count):
        """Test sb run without docker."""
        mocked_failure_count.return_value = 0
        self.cmd('sb run -l localhost -C superbench.enable=none --no-docker', checks=[NoneCheck()])

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

    def test_sb_benchmark_list(self):
        """Test sb benchmark list."""
        self.cmd('sb benchmark list', checks=[JMESPathCheck('length(@)', len(BenchmarkRegistry.benchmarks))])

    def test_sb_benchmark_list_nonexist(self):
        """Test sb benchmark list, give a non-exist benchmark name, should fail."""
        result = self.cmd('sb benchmark list -n non-exist-name', expect_failure=True)
        self.assertEqual(result.exit_code, 1)

    def test_sb_benchmark_list_parameters(self):
        """Test sb benchmark list-parameters."""
        self.cmd('sb benchmark list-parameters', checks=[NoneCheck()])
        self.cmd('sb benchmark list-parameters -n pytorch-[a-z]+', checks=[NoneCheck()])

    def test_sb_node_info(self):
        """Test sb node info, should fail."""
        self.cmd('sb node info', expect_failure=False)

    def test_sb_result_diagnosis(self):
        """Test sb result diagnosis."""
        test_analyzer_dir = str(Path(__file__).parent.resolve() / '../analyzer/')
        # test positive case
        self.cmd(
            'sb result diagnosis -d {dir}/test_results.jsonl -r {dir}/test_rules.yaml -b {dir}/test_baseline.json'.
            format(dir=test_analyzer_dir) + ' --output-dir outputs/test-diagnosis/'
        )
        self.cmd(
            'sb result diagnosis -d {dir}/test_results.jsonl -r {dir}/test_rules.yaml -b {dir}/test_baseline.json'.
            format(dir=test_analyzer_dir) + ' --output-dir outputs/test-diagnosis/ --output-all'
        )
        self.cmd(
            'sb result diagnosis -d {dir}/test_results.jsonl -r {dir}/test_rules_without_baseline.yaml'.
            format(dir=test_analyzer_dir) +
            ' --output-dir outputs/test-diagnosis/ --output-all --output-file-format json'
        )
        # test invalid output format
        self.cmd(
            'sb result diagnosis -d {dir}/test_results.jsonl -r {dir}/test_rules.yaml -b {dir}/test_baseline.json'.
            format(dir=test_analyzer_dir) + ' --output-dir outputs/test-diagnosis/ --output-file-format abb',
            expect_failure=True
        )

    def test_sb_result_summary(self):
        """Test sb result summary."""
        test_analyzer_dir = str(Path(__file__).parent.resolve() / '../analyzer/')
        # test positive case
        self.cmd(
            'sb result summary -d {dir}/test_results.jsonl -r {dir}/test_summary_rules.yaml'.
            format(dir=test_analyzer_dir) + ' --output-dir /tmp/outputs/test-summary/'
        )
        self.cmd(
            'sb result summary -d {dir}/test_results.jsonl -r {dir}/test_summary_rules.yaml'.
            format(dir=test_analyzer_dir) + ' --output-dir /tmp/outputs/test-summary/ --decimal-place-value 4'
        )
        # test invalid output format
        self.cmd(
            'sb result summary -d {dir}/test_results.jsonl -r {dir}/test_rules.yaml'.format(dir=test_analyzer_dir) +
            ' --output-dir /tmp/outputs/test-summary/ --output-file-format abb',
            expect_failure=True
        )
