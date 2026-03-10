# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvbench_base module."""

import unittest
from argparse import Namespace

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import ReturnCode
from superbench.benchmarks.micro_benchmarks.nvbench_base import parse_time_to_us, NvbenchBase


class TestParseTimeToUs(unittest.TestCase):
    """Test class for parse_time_to_us function."""
    def test_parse_microseconds(self):
        """Test parsing microseconds values."""
        self.assertAlmostEqual(parse_time_to_us('123.45 us'), 123.45)
        self.assertAlmostEqual(parse_time_to_us('123.45us'), 123.45)
        self.assertAlmostEqual(parse_time_to_us('0.5 us'), 0.5)

    def test_parse_nanoseconds(self):
        """Test parsing nanoseconds values (converted to us)."""
        self.assertAlmostEqual(parse_time_to_us('1000 ns'), 1.0)
        self.assertAlmostEqual(parse_time_to_us('1000ns'), 1.0)
        self.assertAlmostEqual(parse_time_to_us('500 ns'), 0.5)
        self.assertAlmostEqual(parse_time_to_us('123.456 ns'), 0.123456)

    def test_parse_milliseconds(self):
        """Test parsing milliseconds values (converted to us)."""
        self.assertAlmostEqual(parse_time_to_us('1 ms'), 1000.0)
        self.assertAlmostEqual(parse_time_to_us('1ms'), 1000.0)
        self.assertAlmostEqual(parse_time_to_us('0.5 ms'), 500.0)
        self.assertAlmostEqual(parse_time_to_us('0.001 ms'), 1.0)

    def test_parse_percentage(self):
        """Test parsing percentage values."""
        self.assertAlmostEqual(parse_time_to_us('50.5%'), 50.5)
        self.assertAlmostEqual(parse_time_to_us('0.1%'), 0.1)
        self.assertAlmostEqual(parse_time_to_us('100%'), 100.0)

    def test_parse_plain_number(self):
        """Test parsing plain numbers without unit (defaults to us)."""
        self.assertAlmostEqual(parse_time_to_us('123.45'), 123.45)
        self.assertAlmostEqual(parse_time_to_us('0'), 0.0)

    def test_parse_with_whitespace(self):
        """Test parsing values with leading/trailing whitespace."""
        self.assertAlmostEqual(parse_time_to_us('  123.45 us  '), 123.45)
        self.assertAlmostEqual(parse_time_to_us('\t500 ns\n'), 0.5)

    def test_parse_seconds(self):
        """Test parsing seconds values (converted to us)."""
        self.assertAlmostEqual(parse_time_to_us('1 s'), 1000000.0)
        self.assertAlmostEqual(parse_time_to_us('1s'), 1000000.0)
        self.assertAlmostEqual(parse_time_to_us('0.5 s'), 500000.0)
        self.assertAlmostEqual(parse_time_to_us('0.001 s'), 1000.0)


class ConcreteNvbenchBase(NvbenchBase):
    """Concrete implementation of NvbenchBase for testing."""
    def __init__(self, name, parameters=''):
        """Constructor."""
        super().__init__(name, parameters)
        self._bin_name = 'test_nvbench_binary'


class TestNvbenchBase(BenchmarkTestCase, unittest.TestCase):
    """Test class for NvbenchBase class."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/test_nvbench_binary'])

    def test_nvbench_base_init(self):
        """Test NvbenchBase initialization."""
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='')
        assert benchmark._bin_name == 'test_nvbench_binary'
        assert benchmark.name == 'test-benchmark'

    def test_nvbench_base_add_parser_arguments(self):
        """Test NvbenchBase add_parser_arguments."""
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='')
        benchmark._preprocess()

        # Check default values
        assert benchmark._args.devices is None
        self.assertAlmostEqual(benchmark._args.skip_time, -1.0)
        self.assertAlmostEqual(benchmark._args.throttle_threshold, 75.0)
        self.assertAlmostEqual(benchmark._args.throttle_recovery_delay, 0.05)
        assert benchmark._args.run_once is False
        assert benchmark._args.disable_blocking_kernel is False
        assert benchmark._args.profile is False
        assert benchmark._args.timeout == 15
        assert benchmark._args.min_samples == 10
        assert benchmark._args.stopping_criterion == 'stdrel'
        self.assertAlmostEqual(benchmark._args.min_time, 0.5)
        self.assertAlmostEqual(benchmark._args.max_noise, 0.5)
        self.assertAlmostEqual(benchmark._args.max_angle, 0.048)
        self.assertAlmostEqual(benchmark._args.min_r2, 0.36)

    def test_nvbench_base_preprocess_default(self):
        """Test NvbenchBase preprocess with default parameters."""
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='')
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS
        assert len(benchmark._commands) == 1
        # Check default stopping criterion args are included
        assert '--timeout 15' in benchmark._commands[0]
        assert '--min-samples 10' in benchmark._commands[0]
        assert '--stopping-criterion stdrel' in benchmark._commands[0]
        assert '--min-time 0.5' in benchmark._commands[0]
        assert '--max-noise 0.5' in benchmark._commands[0]

    def test_nvbench_base_preprocess_with_devices(self):
        """Test NvbenchBase preprocess with device configuration."""
        # Test with specific device
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='--devices 0')
        assert benchmark._preprocess()
        assert '--devices 0' in benchmark._commands[0]

        # Test with 'all' devices
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='--devices all')
        assert benchmark._preprocess()
        assert '--devices all' in benchmark._commands[0]

        # Test with multiple devices
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='--devices 0,1,2')
        assert benchmark._preprocess()
        assert '--devices 0,1,2' in benchmark._commands[0]

    def test_nvbench_base_preprocess_with_benchmark_properties(self):
        """Test NvbenchBase preprocess with benchmark properties."""
        parameters = (
            '--skip-time 1.0 '
            '--throttle-threshold 80.0 '
            '--throttle-recovery-delay 0.1 '
            '--run-once '
            '--disable-blocking-kernel '
            '--profile'
        )
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        assert '--skip-time 1.0' in benchmark._commands[0]
        assert '--throttle-threshold 80.0' in benchmark._commands[0]
        assert '--throttle-recovery-delay 0.1' in benchmark._commands[0]
        assert '--run-once' in benchmark._commands[0]
        assert '--disable-blocking-kernel' in benchmark._commands[0]
        assert '--profile' in benchmark._commands[0]

    def test_nvbench_base_preprocess_with_stdrel_stopping_criterion(self):
        """Test NvbenchBase preprocess with stdrel stopping criterion."""
        parameters = (
            '--stopping-criterion stdrel '
            '--min-time 2.0 '
            '--max-noise 0.3 '
            '--timeout 30 '
            '--min-samples 100'
        )
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        assert '--stopping-criterion stdrel' in benchmark._commands[0]
        assert '--min-time 2.0' in benchmark._commands[0]
        assert '--max-noise 0.3' in benchmark._commands[0]
        assert '--timeout 30' in benchmark._commands[0]
        assert '--min-samples 100' in benchmark._commands[0]

    def test_nvbench_base_preprocess_with_entropy_stopping_criterion(self):
        """Test NvbenchBase preprocess with entropy stopping criterion."""
        parameters = (
            '--stopping-criterion entropy '
            '--max-angle 0.1 '
            '--min-r2 0.5 '
            '--timeout 20 '
            '--min-samples 50'
        )
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        assert '--stopping-criterion entropy' in benchmark._commands[0]
        assert '--max-angle 0.1' in benchmark._commands[0]
        assert '--min-r2 0.5' in benchmark._commands[0]
        assert '--timeout 20' in benchmark._commands[0]
        assert '--min-samples 50' in benchmark._commands[0]
        # stdrel args should not be in entropy mode
        assert '--min-time' not in benchmark._commands[0]
        assert '--max-noise' not in benchmark._commands[0]

    def test_nvbench_base_parse_time_value(self):
        """Test NvbenchBase _parse_time_value method."""
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='')
        benchmark._preprocess()

        self.assertAlmostEqual(benchmark._parse_time_value('100 us'), 100.0)
        self.assertAlmostEqual(benchmark._parse_time_value('1000 ns'), 1.0)
        self.assertAlmostEqual(benchmark._parse_time_value('1 ms'), 1000.0)

    def test_nvbench_base_parse_percentage(self):
        """Test NvbenchBase _parse_percentage method."""
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='')
        benchmark._preprocess()

        self.assertAlmostEqual(benchmark._parse_percentage('50.5%'), 50.5)
        self.assertAlmostEqual(benchmark._parse_percentage('100%'), 100.0)
        self.assertAlmostEqual(benchmark._parse_percentage('0.1%'), 0.1)
        self.assertAlmostEqual(benchmark._parse_percentage(25.0), 25.0)

    def test_nvbench_base_handle_parsing_error(self):
        """Test NvbenchBase _handle_parsing_error method."""
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters='')
        benchmark._preprocess()

        benchmark._handle_parsing_error('Test error message', 'raw output data')
        assert benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE

    def test_nvbench_base_build_base_command_without_bin_name(self):
        """Test NvbenchBase _build_base_command raises error without bin_name."""
        benchmark = NvbenchBase('test-benchmark', parameters='')
        # Manually call add_parser_arguments and parse to set up _args
        benchmark.add_parser_arguments()
        benchmark._args = benchmark._parser.parse_args([])
        benchmark._args.bin_dir = '/mock/bin'

        with self.assertRaises(ValueError) as context:
            benchmark._build_base_command()
        assert 'Subclass must set _bin_name' in str(context.exception)

    def test_nvbench_base_full_command_line(self):
        """Test NvbenchBase generates complete command line with all options."""
        parameters = (
            '--devices 0,1 '
            '--skip-time 0.5 '
            '--throttle-threshold 85.0 '
            '--throttle-recovery-delay 0.02 '
            '--run-once '
            '--timeout 60 '
            '--min-samples 200 '
            '--stopping-criterion stdrel '
            '--min-time 1.5 '
            '--max-noise 0.25'
        )
        benchmark = ConcreteNvbenchBase('test-benchmark', parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        cmd = benchmark._commands[0]
        assert 'test_nvbench_binary' in cmd
        assert '--devices 0,1' in cmd
        assert '--skip-time 0.5' in cmd
        assert '--throttle-threshold 85.0' in cmd
        assert '--throttle-recovery-delay 0.02' in cmd
        assert '--run-once' in cmd
        assert '--timeout 60' in cmd
        assert '--min-samples 200' in cmd
        assert '--stopping-criterion stdrel' in cmd
        assert '--min-time 1.5' in cmd
        assert '--max-noise 0.25' in cmd


if __name__ == '__main__':
    unittest.main()
